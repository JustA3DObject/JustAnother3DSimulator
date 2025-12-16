import numpy as np
import os

# PARAMETERS
REMUS_PARAMS = {
    "L": 1.33,
    "D": 0.191,
}

def write_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        f.write(f"# Generated AUV Mesh: {filename}\n")
        f.write(f"o {os.path.basename(filename).split('.')[0]}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join([str(idx) for idx in face]) + "\n")
    print(f"Saved {filename}")

class AUVGeometryFinal:
    def __init__(self):
        self.geo = self._get_geometry_params()
        # Center of Buoyancy offset (to center mesh at 0,0,0)
        self.origin = np.array([0.61, 0.0, 0.0]) 
        self.meshes = {}
        self._generate()

    def _get_geometry_params(self):
        old_L = 1.33
        new_L = REMUS_PARAMS["L"]
        scale = new_L / old_L
        return {
            'a': 0.191 * scale,       
            'a_offset': 0.0165 * scale, 
            'c_offset': 0.0368 * scale, 
            'n': 2,
            'd': REMUS_PARAMS["D"],
            'lf': 0.828 * scale,      
            'l': new_L,               
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
                faces.append([p1, p2, p3, p4])
            p1 = i * cols + cols
            p2 = i * cols + 1
            p3 = (i + 1) * cols + 1
            p4 = (i + 1) * cols + cols
            faces.append([p1, p2, p3, p4])
        return vertices, faces

    def _generate(self):
        geo = self.geo
        
        # HULL
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
        x_profile = np.append(x_profile, geo['l'])
        r_profile = np.append(r_profile, 0.0)
        
        hull_verts, hull_faces = self._generate_revolution_surface(x_profile, r_profile)
        self.meshes['remus_hull.obj'] = (hull_verts, hull_faces)

        # FINS (Added Thickness)
        all_fin_verts = []
        all_fin_faces = []
        fin_offset = 0
        
        # Fin Params
        fin_length = 0.12
        fin_span = 0.1
        fin_taper_ratio = 0.8
        
        fin_x_end = geo['l'] - 0.025
        fin_x_start = fin_x_end - fin_length
        c_len = geo['l'] - geo['lf']
        
        # Calculate Root Radius based on Hull Curvature (Essential for fit)
        x_norm_fin_start = (fin_x_start - geo['lf']) / c_len
        r_fin_start = geo['c_offset'] + ((geo['d']/2) - geo['c_offset']) * (1 - x_norm_fin_start**geo['n'])
        
        x_norm_fin_end = (fin_x_end - geo['lf']) / c_len
        r_fin_end = geo['c_offset'] + ((geo['d']/2) - geo['c_offset']) * (1 - x_norm_fin_end**geo['n'])
        
        # Thickness
        t_half = 0.01 # 2cm total thickness
        
        # Generate 4 fins
        for i in range(4):
            angle = i * (np.pi / 2)
            
            # Vertices for one fin (centered at Z-up initially)
            # We define left side (Y+) and right side (Y-) for thickness
            
            # Root Profile (Follows hull curvature)
            v_root_le = [fin_x_start, 0, r_fin_start - 0.005] # Embed slightly
            v_root_te = [fin_x_end,   0, r_fin_end - 0.005]
            
            # Tip Profile
            v_tip_le = [fin_x_start + 0.02, 0, r_fin_start + fin_span] # Sweep
            v_tip_te = [fin_x_end,          0, r_fin_end + fin_span * fin_taper_ratio]
            
            # 8 Vertices for the box fin
            verts = [
                # Left side (Y = +t_half)
                [v_root_le[0], t_half, v_root_le[2]], # 0
                [v_root_te[0], t_half, v_root_te[2]], # 1
                [v_tip_te[0],  t_half, v_tip_te[2]],  # 2
                [v_tip_le[0],  t_half, v_tip_le[2]],  # 3
                # Right side (Y = -t_half)
                [v_root_le[0], -t_half, v_root_le[2]], # 4
                [v_root_te[0], -t_half, v_root_te[2]], # 5
                [v_tip_te[0],  -t_half, v_tip_te[2]],  # 6
                [v_tip_le[0],  -t_half, v_tip_le[2]],  # 7
            ]
            
            # Faces
            local_faces = [
                [1, 2, 3, 4], [8, 7, 6, 5], # Sides
                [4, 3, 7, 8], [2, 1, 5, 6], # LE/TE
                [3, 2, 6, 7], [1, 4, 8, 5]  # Tip/Root
            ]
            
            # Rotate vertices
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
            
            # Center and Rotate
            v_array = np.array(verts)
            v_array[:, 0] -= self.origin[0] # Center X
            v_rot = (R @ v_array.T).T
            
            all_fin_verts.extend(v_rot.tolist())
            for face in local_faces:
                all_fin_faces.append([idx + fin_offset for idx in face])
            
            fin_offset += 8
            
        self.meshes['remus_fins.obj'] = (all_fin_verts, all_fin_faces)

        # PROPELLER (4 Blades, Attached to Hub)
        # Hub Cylinder
        hub_len = 0.05
        x_hub = np.linspace(geo['l'], geo['l'] + hub_len, 5)
        r_hub = np.full_like(x_hub, geo['c_offset'] * 0.8) 
        x_hub = np.append(x_hub, geo['l'] + hub_len); r_hub = np.append(r_hub, 0.0) # Cap
        hub_verts, hub_faces = self._generate_revolution_surface(x_hub, r_hub, num_theta=20)
        
        # Blades
        blade_verts = []
        blade_faces = []
        offset = len(hub_verts)
        num_blades = 4
        blade_x = geo['l'] + hub_len/2
        
        for i in range(num_blades):
            angle_base = i * (2 * np.pi / num_blades)
            # Simple Blade
            v1 = [blade_x, -0.01, r_hub[0]]; v2 = [blade_x, 0.01, r_hub[0]]
            v3 = [blade_x, 0.01, r_hub[0]+0.08]; v4 = [blade_x, -0.01, r_hub[0]+0.08]
            b_vs = [v1, v2, v3, v4]
            
            R = np.array([[1,0,0],[0,np.cos(angle_base),-np.sin(angle_base)],[0,np.sin(angle_base),np.cos(angle_base)]])
            b_rot = (R @ np.array(b_vs).T).T
            b_rot[:, 0] -= self.origin[0]
            
            blade_verts.extend(b_rot.tolist())
            blade_faces.append([offset+1, offset+2, offset+3, offset+4])
            blade_faces.append([offset+4, offset+3, offset+2, offset+1])
            offset += 4
            
        full_prop_verts = hub_verts + blade_verts
        full_prop_faces = hub_faces + blade_faces
        self.meshes['remus_propeller.obj'] = (full_prop_verts, full_prop_faces)

    def export(self):
        for filename, (v, f) in self.meshes.items():
            write_obj(filename, v, f)

if __name__ == "__main__":
    builder = AUVGeometryFinal()
    builder.export()
