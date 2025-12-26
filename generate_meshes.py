import numpy as np
import os

# PARAMETERS
REMUS_PARAMS = {
    "L": 1.33,
    "D": 0.191,
}

# MATERIAL DEFINITIONS
MATERIALS = {
    "Mat_Nose": (0.0, 0.0, 1.0),   # Blue
    "Mat_Body": (0.0, 1.0, 0.0),   # Green
    "Mat_Tail": (1.0, 0.0, 0.0),   # Red
    "Mat_Fins": (1.0, 1.0, 0.0),   # Yellow
    "Mat_Prop": (0.2, 0.2, 0.2),   # Dark Grey
}

def write_mtl(filename):
    with open(filename, 'w') as f:
        f.write(f"# Generated Library\n")
        for name, rgb in MATERIALS.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {rgb[0]} {rgb[1]} {rgb[2]}\n") 
            f.write("Ka 0.1 0.1 0.1\n") 
            f.write("Ks 0.5 0.5 0.5\n") 
            f.write("Ns 50.0\n")       
            f.write("d 1.0\n\n")        

def write_obj(filename, vertices, parts, mtl_filename="remus.mtl"):
    with open(filename, 'w') as f:
        f.write(f"# Generated AUV Mesh: {filename}\n")
        f.write(f"mtllib {mtl_filename}\n") 
        f.write(f"o {os.path.basename(filename).split('.')[0]}\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        for mat_name, faces in parts.items():
            f.write(f"usemtl {mat_name}\n")
            for face in faces:
                f.write("f " + " ".join([str(idx) for idx in face]) + "\n")
    
    print(f"Saved {filename}")

class AUVGeometryFinal:
    def __init__(self):
        self.geo = self._get_geometry_params()
        self.origin = np.array([0.61, 0.0, 0.0]) 
        self.meshes = {} 
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
        
        for i in range(rows - 1):
            for j in range(cols - 1):
                p1 = i * cols + j + 1
                p2 = i * cols + (j + 1) + 1
                p3 = (i + 1) * cols + (j + 1) + 1
                p4 = (i + 1) * cols + j + 1
                faces.append([p4, p3, p2, p1])
            
            p1 = i * cols + cols
            p2 = i * cols + 1
            p3 = (i + 1) * cols + 1
            p4 = (i + 1) * cols + cols
            faces.append([p4, p3, p2, p1])
            
        return vertices, faces

    def _generate_prism_fin(self, root_x, root_z, tip_x, tip_z, thickness):
        t_half = thickness / 2
        
        verts = [
            [root_x[0], 0.0, root_z[0]],       # 1: Root LE
            [tip_x[0],  0.0, tip_z[0]],        # 2: Tip LE
            [root_x[1], t_half, root_z[1]],    # 3: Root TE Top
            [root_x[1], -t_half, root_z[1]],   # 4: Root TE Bot
            [tip_x[1],  t_half, tip_z[1]],     # 5: Tip TE Top
            [tip_x[1],  -t_half, tip_z[1]]     # 6: Tip TE Bot
        ]
        
        verts = np.array(verts)
        verts[:, 0] -= self.origin[0]
        
        faces = [
            [1, 3, 5, 2], # Top Slope
            [1, 2, 6, 4], # Bot Slope
            [3, 4, 6, 5], # Back Face
            [1, 4, 3],    # Root Side
            [2, 5, 6]     # Tip Side
        ]
        return verts.tolist(), faces

    def _generate_twisted_blade(self, hub_r, blade_len, center_x):
        """
        Generates volumetric blade.
        """
        # BLADE PARAMETERS
        root_chord = 0.035
        tip_chord = 0.02
        root_thick = 0.006
        tip_thick = 0.002
        root_pitch = np.radians(45) # Steep angle at root
        tip_pitch = np.radians(20)  # Shallow angle at tip
        skew = 0.01 # Sweep back distance

        # Define 4-point airfoil section (Diamond shape) relative to section center
        # Points: LE, Top, TE, Bot
        def get_section_verts(chord, thick, pitch, r_offset, x_skew):
            # Unrotated 2D coords (x=axial, y=tangential/thickness)
            # Centered at roughly 30% chord
            le_x = -0.3 * chord
            te_x = 0.7 * chord
            
            # Basic Diamond Profile
            pts = np.array([
                [le_x, 0.0, r_offset],         # LE
                [0.0, thick/2, r_offset],      # Top
                [te_x, 0.0, r_offset],         # TE
                [0.0, -thick/2, r_offset]      # Bot
            ])
            
            # Apply Pitch Rotation (around Radial Z-axis of the blade)
            # In local frame: X is axial, Y is tangential. Rotate X/Y.
            c, s = np.cos(pitch), np.sin(pitch)
            rot_mat = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            
            rotated = (rot_mat @ pts.T).T
            
            # Apply Global Offset (Hub center X + skew)
            rotated[:, 0] += center_x + x_skew
            return rotated

        # Generate Root and Tip sections
        root_verts = get_section_verts(root_chord, root_thick, root_pitch, hub_r, 0.0)
        tip_verts = get_section_verts(tip_chord, tip_thick, tip_pitch, hub_r + blade_len, skew)
        
        all_verts = np.vstack([root_verts, tip_verts])
        all_verts[:, 0] -= self.origin[0] # Adjust to global origin

        # Faces (Lofting Root to Tip)
        # Root indices: 0-3, Tip indices: 4-7
        # Order: LE(0), Top(1), TE(2), Bot(3)
        faces = [
            [0, 4, 5, 1], # Top-Front slope
            [1, 5, 6, 2], # Top-Back slope
            [2, 6, 7, 3], # Bot-Back slope
            [3, 7, 4, 0], # Bot-Front slope
            [4, 7, 6, 5]  # Tip Cap (Diamond)
            # Root cap is hidden inside hub, no need to draw
        ]
        
        return all_verts.tolist(), faces

    def _generate(self):
        geo = self.geo
        
        # HULL GENERATION
        num_points = 50
        x_nose = np.linspace(0, geo['a'], num_points)
        r_nose = (geo['d']/2) - ((geo['d']/2) - geo['a_offset']) * (1 - x_nose/geo['a'])**geo['n']
        x_mid = np.linspace(geo['a'], geo['lf'], num_points)
        r_mid = np.full_like(x_mid, geo['d']/2)
        x_tail = np.linspace(geo['lf'], geo['l'], num_points)
        c_len = geo['l'] - geo['lf']
        r_tail = geo['c_offset'] + ((geo['d']/2) - geo['c_offset']) * (1 - ((x_tail - geo['lf'])/c_len)**geo['n'])
        
        x_profile = np.concatenate([x_nose, x_mid[1:], x_tail[1:]])
        r_profile = np.concatenate([r_nose, r_mid[1:], r_tail[1:]])
        x_profile = np.insert(x_profile, 0, 0.0); r_profile = np.insert(r_profile, 0, 0.001)
        x_profile = np.append(x_profile, geo['l']); r_profile = np.append(r_profile, 0.001)
        
        hull_verts, hull_faces = self._generate_revolution_surface(x_profile, r_profile)
        
        faces_per_row = 40 
        split_idx_1 = 50 * faces_per_row
        split_idx_2 = 99 * faces_per_row
        
        self.meshes['remus_hull.obj'] = (hull_verts, {
            "Mat_Nose": hull_faces[:split_idx_1],
            "Mat_Body": hull_faces[split_idx_1:split_idx_2],
            "Mat_Tail": hull_faces[split_idx_2:]
        })

        # FIN GENERATION
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

        # PROPELLER GENERATION 
        # Hub
        x_hub = np.linspace(geo['l'], geo['l'] + 0.05, 5)
        r_hub_val = geo['c_offset'] * 0.8
        r_hub = np.full_like(x_hub, r_hub_val)
        x_hub = np.append(x_hub, geo['l'] + 0.05); r_hub = np.append(r_hub, 0.001)
        hub_verts, hub_faces = self._generate_revolution_surface(x_hub, r_hub, num_theta=20)
        
        # Blades
        blade_verts = []; blade_faces = []; offset = len(hub_verts)
        num_blades = 3 # 3 blades is common for REMUS, but can be 4
        blade_len = 0.09
        blade_x_center = geo['l'] + 0.02
        
        for i in range(num_blades):
            angle = i * (2 * np.pi / num_blades)
            
            # Generate straight blade vertices (pointing up Z)
            v_b, f_b = self._generate_twisted_blade(r_hub_val, blade_len, blade_x_center)
            
            # Rotate blade around X axis to position
            R = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
            # v_b is already adjusted to global origin, but we need to rotate it relative to the prop axis (which is X axis)
            # The function returned coordinates assuming X is axis. We just need to apply the roll.
            
            # Note: v_b includes the 'self.origin' subtraction. We must add it back for rotation, or rotate the relative vector.
            # v_b are vectors relative to origin (0,0,0) in the file space.
            # Let's treat them as vectors relative to axis for rotation components Y and Z.
            
            v_b_arr = np.array(v_b)
            # Y and Z are the cross-section components. X is axial.
            # We rotate Y and Z around X.
            v_rot = (R @ v_b_arr.T).T
            
            blade_verts.extend(v_rot.tolist())
            
            for face in f_b:
                blade_faces.append([idx + offset for idx in face])
            
            offset += 8 # 8 vertices per blade (4 root + 4 tip)
            
        self.meshes['remus_propeller.obj'] = (hub_verts + blade_verts, {
            "Mat_Prop": hub_faces + blade_faces
        })

    def export(self):
        write_mtl("remus.mtl")
        for filename, (v, parts) in self.meshes.items():
            write_obj(filename, v, parts)

if __name__ == "__main__":
    AUVGeometryFinal().export()
