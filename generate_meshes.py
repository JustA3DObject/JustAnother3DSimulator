import numpy as np
import os

# PARAMETERS
REMUS_PARAMS = {
    "L": 1.33,
    "D": 0.191,
}

# MATERIAL DEFINITIONS
# Defines the colors: Name -> (R, G, B)
MATERIALS = {
    "Mat_Nose": (0.0, 0.0, 1.0),   # Blue
    "Mat_Body": (0.0, 1.0, 0.0),   # Green
    "Mat_Tail": (1.0, 0.0, 0.0),   # Red
    "Mat_Fins": (1.0, 1.0, 0.0),   # Yellow
    "Mat_Prop": (0.1, 0.1, 0.1),   # Black (Dark Grey for visibility)
}

def write_mtl(filename):
    """Writes the material library file."""
    with open(filename, 'w') as f:
        f.write(f"# Generated Material Library\n")
        for name, rgb in MATERIALS.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {rgb[0]} {rgb[1]} {rgb[2]}\n") # Diffuse color
            f.write("Ka 0.1 0.1 0.1\n") # Ambient
            f.write("Ks 0.5 0.5 0.5\n") # Specular
            f.write("Ns 50.0\n")        # Shininess
            f.write("d 1.0\n\n")        # Opacity

def write_obj(filename, vertices, parts, mtl_filename="remus.mtl"):
    """
    Writes an OBJ file with material references.
    'parts' is a dictionary: { "MaterialName": [list_of_faces], ... }
    """
    with open(filename, 'w') as f:
        f.write(f"# Generated AUV Mesh: {filename}\n")
        f.write(f"mtllib {mtl_filename}\n") # Reference the material file
        f.write(f"o {os.path.basename(filename).split('.')[0]}\n")
        
        # Write all vertices first
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces grouped by material
        for mat_name, faces in parts.items():
            f.write(f"usemtl {mat_name}\n")
            for face in faces:
                f.write("f " + " ".join([str(idx) for idx in face]) + "\n")
    
    print(f"Saved {filename}")

class AUVGeometryFinal:
    def __init__(self):
        self.geo = self._get_geometry_params()
        self.origin = np.array([0.61, 0.0, 0.0]) 
        self.meshes = {} # Format: 'filename': (vertices, {mat: faces, ...})
        self._generate()

    def _get_geometry_params(self):
        scale = REMUS_PARAMS["L"] / 1.33
        return {
            'a': 0.191 * scale,       
            'a_offset': 0.0165 * scale, 
            'c_offset': 0.0368 * scale, 
            'n': 2,
            'd': REMUS_PARAMS["D"],
            'lf': 0.828 * scale,      
            'l': REMUS_PARAMS["L"],               
        }

    def _generate_revolution_surface(self, x_profile, r_profile, num_theta=40):
        theta = np.linspace(0, 2 * np.pi, num_theta)
        vertices = []
        faces = []
        rows = len(x_profile)
        cols = len(theta)
        
        for i in range(rows):
            x = x_profile[i]
            r = r_profile[i]
            for th in theta:
                y = r * np.cos(th)
                z = r * np.sin(th)
                vertices.append([x - self.origin[0], y, z])
        
        # Generate Faces (Quads)
        for i in range(rows - 1):
            for j in range(cols - 1):
                p1 = i * cols + j + 1
                p2 = i * cols + (j + 1) + 1
                p3 = (i + 1) * cols + (j + 1) + 1
                p4 = (i + 1) * cols + j + 1
                faces.append([p4, p3, p2, p1])
                
            # Close the loop (seam)
            p1 = i * cols + cols
            p2 = i * cols + 1
            p3 = (i + 1) * cols + 1
            p4 = (i + 1) * cols + cols
            faces.append([p4, p3, p2, p1])
            
        return vertices, faces

    def _generate_prism_fin(self, root_x, root_z, tip_x, tip_z, thickness):
        """
        Generates a triangular prism fin.
        Pointed side towards nose.
        Square side towards tail.
        """
        t_half = thickness / 2
        
        # 6 Vertices for a triangular prism
        # Root Leading Edge (Center - Pointed)
        # Tip Leading Edge (Center - Pointed)
        # Root Trailing Edge (Top)
        # Root Trailing Edge (Bottom)
        # Tip Trailing Edge (Top)
        # Tip Trailing Edge (Bottom)
        
        verts = [
            [root_x[0], 0.0, root_z[0]],       # Root LE
            [tip_x[0],  0.0, tip_z[0]],        # Tip LE
            [root_x[1], t_half, root_z[1]],    # Root TE Top
            [root_x[1], -t_half, root_z[1]],   # Root TE Bot
            [tip_x[1],  t_half, tip_z[1]],     # Tip TE Top
            [tip_x[1],  -t_half, tip_z[1]]     # Tip TE Bot
        ]
        
        verts = np.array(verts)
        verts[:, 0] -= self.origin[0]
        
        # Define Faces (1-based indices relative to this specific fin's vertex list)
        # Ensure CCW winding order for outside normals
        faces = [
            # Top Slope (Quad): Root LE -> Root Top -> Tip Top -> Tip LE
            [1, 3, 5, 2],
            # Bottom Slope (Quad): Root LE -> Tip LE -> Tip Bot -> Root Bot
            [1, 2, 6, 4],
            # Back Face (Quad - "Square side"): Root Top -> Root Bot -> Tip Bot -> Tip Top
            [3, 4, 6, 5],
            # Root Side (Triangle): Root LE -> Root Bot -> Root Top
            [1, 4, 3],
            # Tip Side (Triangle): Tip LE -> Tip Top -> Tip Bot
            [2, 5, 6]
        ]
        
        return verts.tolist(), faces

    def _generate(self):
        geo = self.geo
        
        # HULL
        num_points = 50
        # Generate profiles
        x_nose = np.linspace(0, geo['a'], num_points)
        r_nose = (geo['d']/2) - ((geo['d']/2) - geo['a_offset']) * (1 - x_nose/geo['a'])**geo['n']
        
        x_mid = np.linspace(geo['a'], geo['lf'], num_points)
        r_mid = np.full_like(x_mid, geo['d']/2)
        
        x_tail = np.linspace(geo['lf'], geo['l'], num_points)
        c_len = geo['l'] - geo['lf']
        r_tail = geo['c_offset'] + ((geo['d']/2) - geo['c_offset']) * (1 - ((x_tail - geo['lf'])/c_len)**geo['n'])
        
        x_profile = np.concatenate([x_nose, x_mid[1:], x_tail[1:]])
        r_profile = np.concatenate([r_nose, r_mid[1:], r_tail[1:]])
        
        # Add Caps
        x_profile = np.insert(x_profile, 0, 0.0); r_profile = np.insert(r_profile, 0, 0.001)
        x_profile = np.append(x_profile, geo['l']); r_profile = np.append(r_profile, 0.001)
        
        hull_verts, hull_faces = self._generate_revolution_surface(x_profile, r_profile)
        
        # Calculate Face Splits
        num_theta = 40
        faces_per_row = num_theta 
        row_split_1 = 50 
        row_split_2 = 99 
        split_idx_1 = row_split_1 * faces_per_row
        split_idx_2 = row_split_2 * faces_per_row
        
        faces_nose = hull_faces[:split_idx_1]
        faces_body = hull_faces[split_idx_1:split_idx_2]
        faces_tail = hull_faces[split_idx_2:]
        
        self.meshes['remus_hull.obj'] = (hull_verts, {
            "Mat_Nose": faces_nose,
            "Mat_Body": faces_body,
            "Mat_Tail": faces_tail
        })

        # FINS
        all_fin_verts = []; all_fin_faces = []; fin_offset = 0
        fin_length = 0.12; fin_span = 0.1; fin_taper_ratio = 0.8
        fin_x_end = geo['l'] - 0.025; fin_x_start = fin_x_end - fin_length
        c_len = geo['l'] - geo['lf']
        r_fin_start = geo['c_offset'] + ((geo['d']/2) - geo['c_offset']) * (1 - ((fin_x_start - geo['lf'])/c_len)**geo['n'])
        r_fin_end = geo['c_offset'] + ((geo['d']/2) - geo['c_offset']) * (1 - ((fin_x_end - geo['lf'])/c_len)**geo['n'])
        fin_span = 0.1
        
        for i in range(4):
            angle = i * (np.pi / 2)
            v_loc, f_loc = self._generate_prism_fin(
                root_x=[fin_x_start, fin_x_end], root_z=[r_fin_start - 0.005, r_fin_end - 0.005],
                tip_x=[fin_x_start + 0.02, fin_x_end], tip_z=[r_fin_start + fin_span, r_fin_end + fin_span * fin_taper_ratio], thickness=0.02
            )
            R = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            v_rot = (R @ np.array(v_loc).T).T
            all_fin_verts.extend(v_rot.tolist())
            for face in f_loc: all_fin_faces.append([idx + fin_offset for idx in face])
            
            fin_offset += 6
            
        self.meshes['remus_fins.obj'] = (all_fin_verts, {
            "Mat_Fins": all_fin_faces
        })

        # PROPELLER
        x_hub = np.linspace(geo['l'], geo['l'] + 0.05, 5)
        r_hub = np.full_like(x_hub, geo['c_offset'] * 0.8)
        x_hub = np.append(x_hub, geo['l'] + 0.05); r_hub = np.append(r_hub, 0.001)
        hub_verts, hub_faces = self._generate_revolution_surface(x_hub, r_hub, num_theta=20)
        
        blade_verts = []; blade_faces = []; offset = len(hub_verts)
        num_blades = 4; blade_x = geo['l'] + 0.025
        for i in range(num_blades):
            angle = i * (2 * np.pi / num_blades)
            v_b = [[blade_x, -0.01, r_hub[0]], [blade_x, 0.01, r_hub[0]], [blade_x, 0.01, r_hub[0]+0.08], [blade_x, -0.01, r_hub[0]+0.08]]
            R = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
            b_rot = (R @ np.array(v_b).T).T; b_rot[:, 0] -= self.origin[0]
            blade_verts.extend(b_rot.tolist())
            blade_faces.append([offset+1, offset+2, offset+3, offset+4])
            blade_faces.append([offset+4, offset+3, offset+2, offset+1])
            offset += 4
            
        self.meshes['remus_propeller.obj'] = (hub_verts + blade_verts, {
            "Mat_Prop": hub_faces + blade_faces
        })

    def export(self):
        # Create the Material Library
        write_mtl("remus.mtl")
        
        # Export OBJs with references to it
        for filename, (v, parts) in self.meshes.items():
            write_obj(filename, v, parts)

if __name__ == "__main__":
    AUVGeometryFinal().export()