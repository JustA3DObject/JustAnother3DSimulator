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
            # OBJ indices are 1-based
            f.write("f " + " ".join([str(idx) for idx in face]) + "\n")
    print(f"Saved {filename}")

class AUVGeometry:
    def __init__(self):
        self.geo = self._get_geometry_params()
        # Center of Buoyancy offset (approx middle of vehicle)
        self.origin_shift = np.array([0.61, 0.0, 0.0]) 
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
                # Shift by origin here
                vertices.append([x - self.origin_shift[0], y, z])
        
        # Generate Faces with Standard CCW Winding (Fixes lighting)
        for i in range(rows - 1):
            for j in range(cols - 1):
                p1 = i * cols + j + 1
                p2 = i * cols + (j + 1) + 1
                p3 = (i + 1) * cols + (j + 1) + 1
                p4 = (i + 1) * cols + j + 1
                faces.append([p1, p2, p3, p4])
                
            # Close the loop
            p1 = i * cols + cols
            p2 = i * cols + 1
            p3 = (i + 1) * cols + 1
            p4 = (i + 1) * cols + cols
            faces.append([p1, p2, p3, p4])
            
        return vertices, faces

    def _generate_box_fin(self, root_x, root_z, tip_x, tip_z, thickness):
        t_half = thickness / 2
        
        # Vertices relative to local fin origin
        # Note: We apply origin_shift later during rotation
        
        verts = [
            # Left side (Y = +t_half)
            [root_x[0], t_half, root_z[0]], # 0: Root LE
            [root_x[1], t_half, root_z[1]], # 1: Root TE
            [tip_x[1],  t_half, tip_z[1]],  # 2: Tip TE
            [tip_x[0],  t_half, tip_z[0]],  # 3: Tip LE
            
            # Right side (Y = -t_half)
            [root_x[0], -t_half, root_z[0]], # 4: Root LE
            [root_x[1], -t_half, root_z[1]], # 5: Root TE
            [tip_x[1],  -t_half, tip_z[1]],  # 6: Tip TE
            [tip_x[0],  -t_half, tip_z[0]],  # 7: Tip LE
        ]
        
        # Faces (CCW winding for outside normals)
        faces = [
            [1, 4, 3, 2], # Left Side (Top if flat)
            [5, 6, 7, 8], # Right Side
            [1, 2, 6, 5], # Root Face
            [3, 4, 8, 7], # Tip Face
            [2, 3, 7, 6], # Trailing Edge
            [4, 1, 5, 8], # Leading Edge
        ]
        
        return verts, faces

    def _generate(self):
        geo = self.geo
        
        # HULL
        num_points = 60
        x_nose = np.linspace(0, geo['a'], num_points)
        r_nose = (geo['d']/2) - ((geo['d']/2) - geo['a_offset']) * (1 - x_nose/geo['a'])**geo['n']
        
        x_mid = np.linspace(geo['a'], geo['lf'], num_points)
        r_mid = np.full_like(x_mid, geo['d']/2)
        
        x_tail = np.linspace(geo['lf'], geo['l'], num_points)
        c_len = geo['l'] - geo['lf']
        r_tail = geo['c_offset'] + ((geo['d']/2) - geo['c_offset']) * (1 - ((x_tail - geo['lf'])/c_len)**geo['n'])
        
        # Combine
        x_profile = np.concatenate([x_nose, x_mid[1:], x_tail[1:]])
        r_profile = np.concatenate([r_nose, r_mid[1:], r_tail[1:]])
        
        # Cap the end
        x_profile = np.append(x_profile, geo['l'])
        r_profile = np.append(r_profile, 0.0)
        
        hull_verts, hull_faces = self._generate_revolution_surface(x_profile, r_profile)
        self.meshes['remus_hull.obj'] = (hull_verts, hull_faces)

        # FINS
        all_fin_verts = []
        all_fin_faces = []
        fin_offset = 0
        
        # HARD CONSTRAINT: Fins start well before the tail ends
        # Hull ends at geo['l'] (1.33m)
        # Fins are 12cm long. Place them 5cm from the very tip.
        fin_end_x = geo['l'] - 0.05 
        fin_start_x = fin_end_x - 0.12
        
        # Calculate exact hull radius at fin location to ensure attachment
        # Using the tail curve formula
        def get_r_at_x(x):
            if x < geo['lf']: return geo['d']/2
            x_norm = (x - geo['lf']) / c_len
            return geo['c_offset'] + ((geo['d']/2) - geo['c_offset']) * (1 - x_norm**geo['n'])

        r_root_start = get_r_at_x(fin_start_x)
        r_root_end = get_r_at_x(fin_end_x)
        
        # Embed fins slightly (minus 1cm) to ensure no gap
        r_root_start -= 0.01
        r_root_end -= 0.01
        
        fin_span = 0.1
        thickness = 0.015 # 1.5cm thick
        
        for i in range(4):
            angle = i * (np.pi / 2)
            
            # Generate local box fin
            v_loc, f_loc = self._generate_box_fin(
                root_x=[fin_start_x, fin_end_x], root_z=[r_root_start, r_root_end],
                tip_x=[fin_start_x + 0.03, fin_end_x], tip_z=[r_root_start + fin_span, r_root_end + fin_span*0.8],
                thickness=thickness
            )
            
            # Rotate and Shift
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
            
            v_rot = (R @ np.array(v_loc).T).T
            v_rot[:, 0] -= self.origin_shift[0] # Shift X to align with hull
            
            all_fin_verts.extend(v_rot.tolist())
            for face in f_loc:
                all_fin_faces.append([idx + fin_offset for idx in face])
            fin_offset += 8
            
        self.meshes['remus_fins.obj'] = (all_fin_verts, all_fin_faces)

        # PROPELLER
        # Hub starts exactly where fins end
        hub_start = geo['l'] - 0.02 # Slight overlap
        hub_len = 0.06
        x_hub = np.linspace(hub_start, hub_start + hub_len, 5)
        # Hub radius matches tail end radius
        r_hub_val = geo['c_offset']
        r_hub = np.full_like(x_hub, r_hub_val)
        
        # Cap hub
        x_hub = np.append(x_hub, hub_start + hub_len)
        r_hub = np.append(r_hub, 0.0)
        
        hub_verts, hub_faces = self._generate_revolution_surface(x_hub, r_hub, num_theta=20)
        
        # Blades
        blade_verts = []
        blade_faces = []
        offset = len(hub_verts)
        num_blades = 4 # You requested 4 blades
        blade_x = hub_start + hub_len/2
        
        for i in range(num_blades):
            angle_base = i * (2 * np.pi / num_blades)
            
            # Simple Blade geometry (thin box)
            v_loc, f_loc = self._generate_box_fin(
                root_x=[blade_x-0.01, blade_x+0.01], root_z=[r_hub_val, r_hub_val],
                tip_x=[blade_x-0.01, blade_x+0.01], tip_z=[r_hub_val+0.08, r_hub_val+0.08],
                thickness=0.005
            )
            
            # Twist the blade 30 degrees
            twist = np.radians(30)
            R_twist = np.array([
                [np.cos(twist), 0, np.sin(twist)],
                [0, 1, 0],
                [-np.sin(twist), 0, np.cos(twist)]
            ])
            # (Simplified twist rotation logic would go here, omitting for stability)

            # Rotate around Hub
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle_base), -np.sin(angle_base)],
                [0, np.sin(angle_base), np.cos(angle_base)]
            ])
            
            v_rot = (R @ np.array(v_loc).T).T
            v_rot[:, 0] -= self.origin_shift[0]
            
            blade_verts.extend(v_rot.tolist())
            for face in f_loc:
                blade_faces.append([idx + offset for idx in face])
            offset += 8
            
        full_prop_verts = hub_verts + blade_verts
        full_prop_faces = hub_faces + blade_faces
        self.meshes['remus_propeller.obj'] = (full_prop_verts, full_prop_faces)

    def export(self):
        for filename, (v, f) in self.meshes.items():
            write_obj(filename, v, f)

if __name__ == "__main__":
    builder = AUVGeometry()
    builder.export()
