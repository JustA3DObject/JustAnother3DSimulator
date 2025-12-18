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

    def _generate_box_fin(self, root_x, root_z, tip_x, tip_z, thickness):
        t_half = thickness / 2
        v_root_le = [root_x[0], 0, root_z[0]]; v_root_te = [root_x[1], 0, root_z[1]]
        v_tip_le = [tip_x[0], 0, tip_z[0]]; v_tip_te = [tip_x[1], 0, tip_z[1]]
        
        verts = [
            [v_root_le[0], t_half, v_root_le[2]], [v_root_te[0], t_half, v_root_te[2]],
            [v_tip_te[0],  t_half, v_tip_te[2]],  [v_tip_le[0],  t_half, v_tip_te[2]],
            [v_root_le[0], -t_half, v_root_le[2]], [v_root_te[0], -t_half, v_root_te[2]],
            [v_tip_te[0],  -t_half, v_tip_te[2]],  [v_tip_le[0],  -t_half, v_tip_le[2]],
        ]
        verts = np.array(verts); verts[:, 0] -= self.origin[0]
        
        faces = [[1, 2, 3, 4], [8, 7, 6, 5], [4, 3, 7, 8], [2, 1, 5, 6], [3, 2, 6, 7], [1, 4, 8, 5]]
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
        
        # Combine profiles
        # We preserve the segmentation logic here to map materials later
        # Original logic: Nose(50) + Mid(49) + Tail(49) + Caps
        
        x_profile = np.concatenate([x_nose, x_mid[1:], x_tail[1:]])
        r_profile = np.concatenate([r_nose, r_mid[1:], r_tail[1:]])
        
        # Add Caps
        x_profile = np.insert(x_profile, 0, 0.0); r_profile = np.insert(r_profile, 0, 0.001)
        x_profile = np.append(x_profile, geo['l']); r_profile = np.append(r_profile, 0.001)
        
        hull_verts, hull_faces = self._generate_revolution_surface(x_profile, r_profile)
        
        # Calculate Face Splits based on point counts
        # Index 0 is Start Cap. 
        # Nose section: Cap(1) + Nose(50) = 51 points -> rows 0 to 50
        # Body section: + Mid(49) = 100 points -> rows 51 to 99
        # Tail section: + Tail(49) + Cap(1) = 150 points -> rows 100 to 149
        
        # Faces are generated per row. There are `num_theta` faces per row segment.
        # We need to slice the list of faces based on rows.
        num_theta = 40
        faces_per_row = num_theta # In revolution surface loop
        
        # Row indices where sections end
        # Nose ends at index 50 (x_nose[-1]). So faces 0 to 49 are Nose.
        row_split_1 = 50 
        # Body ends at index 99 (x_mid[-1]). So faces 50 to 98 are Body.
        row_split_2 = 99 
        
        # Convert row indices to face list indices
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
            v_loc, f_loc = self._generate_box_fin(
                root_x=[fin_x_start, fin_x_end], root_z=[r_fin_start - 0.005, r_fin_end - 0.005],
                tip_x=[fin_x_start + 0.02, fin_x_end], tip_z=[r_fin_start + fin_span, r_fin_end + fin_span * fin_taper_ratio], thickness=0.02
            )
            R = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            v_rot = (R @ np.array(v_loc).T).T
            all_fin_verts.extend(v_rot.tolist())
            for face in f_loc: all_fin_faces.append([idx + fin_offset for idx in face])
            fin_offset += 8
            
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
