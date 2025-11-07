import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from auv_parameters import REMUS_PARAMS, PARAMS_DERIVED
from matplotlib.animation import FuncAnimation

def create_sphere_marker(center, radius, resolution=10):
    """Helper function to create (X, Y, Z) for a sphere surface."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    X = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    Y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    Z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return X, Y, Z

class AUVController: 
    """Controller to make the AUV interactive"""
    def __init__(self, geometry):
        # Store geometry parameters
        self.geometry = geometry

        # Initialize state vars
        self.position = np.array([0.0, 0.0, 0.0]) # [x, y ,z]
        self.orientation = np.array([0.0, 0.0, 0.0]) # [roll, pitch, yaw]
        # Roll control will not be added because AUVs don't have one.

        # Movement parameters
        self.velocity = 0.0 # m/s
        self.max_velocity = 2.0 # m/s
        self.acceleration = 0.5 # m/s^2
        self.deceleration = 0.8 # m/s^2
        self.friction = 0.2 # m/s^2 (Decelerates the AUV and brings it to rest if it has a velocity but no input command)
        self.angular_speed = 0.03 # rad/s
        self.dt = 0.05 # s

        # Key tracking
        self.keys_pressed = set()

        # Generating geometry
        self.generate_base_geometry()


    def generate_base_geometry(self):
        """Generate the AUV geometry"""
        geo = self.geometry
        
        r_max = geo['d'] / 2
        num_x_points = 100
        num_theta_points = 80
        theta = np.linspace(0, 2 * np.pi, num_theta_points)
        z_offset = 0.001
        
        # NOSE SECTION (Elipsoid)
        x_nose = np.linspace(0, geo['a'], num_x_points)
        r_nose = r_max - (r_max - geo['a_offset']) * (1 - x_nose / geo['a'])**geo['n']
        
        X_nose, THETA_nose = np.meshgrid(x_nose, theta)
        R_nose, _ = np.meshgrid(r_nose, theta)
        Y_nose = R_nose * np.cos(THETA_nose)
        Z_nose = R_nose * np.sin(THETA_nose)

        # MID-SECTION (Cylinder)
        mid_section_length = geo['lf'] - geo['a']
        num_x_mid_points = max(2, int(num_x_points * (mid_section_length / geo['a'])))
        x_mid = np.linspace(geo['a'], geo['lf'], num_x_mid_points)
        r_mid = np.full_like(x_mid, r_max)
        
        X_mid, THETA_mid = np.meshgrid(x_mid, theta)
        R_mid, _ = np.meshgrid(r_mid, theta)
        Y_mid = R_mid * np.cos(THETA_mid)
        Z_mid = R_mid * np.sin(THETA_mid)

        # TAIL SECTION (Power Series Curve of Revolution)
        c = geo['l'] - geo['lf']
        num_x_tail_points = max(2, int(num_x_points * (c / geo['a'])))
        x_tail = np.linspace(geo['lf'], geo['l'], num_x_tail_points)
        
        x_norm_tail = (x_tail - geo['lf']) / c
        r_tail = geo['c_offset'] + (r_max - geo['c_offset']) * (1 - x_norm_tail**geo['n'])
        
        X_tail, THETA_tail = np.meshgrid(x_tail, theta)
        R_tail, _ = np.meshgrid(r_tail, theta)
        Y_tail = R_tail * np.cos(THETA_tail)
        Z_tail = R_tail * np.sin(THETA_tail)
        
        r_final = geo['c_offset']

        # Side-Scan Sonar (SSS) Patches 
        sss_len = 0.3
        sss_width_angle = 0.1
        x_sss_start = geo['a'] + (mid_section_length - sss_len) / 2
        x_sss_end = x_sss_start + sss_len
        x_sss = np.linspace(x_sss_start, x_sss_end, 10)
        
        th_sss1 = np.linspace(np.pi - sss_width_angle, np.pi + sss_width_angle, 10)
        X_sss1, TH_sss1 = np.meshgrid(x_sss, th_sss1)
        R_sss1 = r_max + z_offset
        Y_sss1 = R_sss1 * np.cos(TH_sss1)
        Z_sss1 = R_sss1 * np.sin(TH_sss1)
        
        th_sss2 = np.linspace(-sss_width_angle, sss_width_angle, 10)
        X_sss2, TH_sss2 = np.meshgrid(x_sss, th_sss2)
        R_sss2 = r_max + z_offset
        Y_sss2 = R_sss2 * np.cos(TH_sss2)
        Z_sss2 = R_sss2 * np.sin(TH_sss2)

        # DVL (Doppler Velocity Log) Box
        dvl_len = 0.1
        dvl_width = 0.08
        dvl_height = 0.03
        x_dvl_start = geo['lf'] - dvl_len - 0.05
        x_dvl_end = x_dvl_start + dvl_len
        y_dvl_half = dvl_width / 2
        z_dvl_top = -r_max
        z_dvl_bottom = z_dvl_top - dvl_height
        v = np.array([
            [x_dvl_start, -y_dvl_half, z_dvl_top], [x_dvl_end, -y_dvl_half, z_dvl_top],
            [x_dvl_end, y_dvl_half, z_dvl_top], [x_dvl_start, y_dvl_half, z_dvl_top],
            [x_dvl_start, -y_dvl_half, z_dvl_bottom], [x_dvl_end, -y_dvl_half, z_dvl_bottom],
            [x_dvl_end, y_dvl_half, z_dvl_bottom], [x_dvl_start, y_dvl_half, z_dvl_bottom]
        ])
        dvl_faces = [
            [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], [v[0], v[1], v[5], v[4]],
            [v[2], v[3], v[7], v[6]], [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]]
        ]

        # Antenna Mast
        mast_height = 0.08
        mast_radius = 0.01
        x_mast_base = geo['lf'] - 0.1
        theta_mast = np.linspace(0, 2 * np.pi, 20)
        z_mast = np.linspace(r_max, r_max + mast_height, 2)
        TH_mast, Z_mast = np.meshgrid(theta_mast, z_mast)
        X_mast = x_mast_base + mast_radius * np.cos(TH_mast)
        Y_mast = 0 + mast_radius * np.sin(TH_mast)
        
        # Fins
        fin_length = 0.12
        fin_span = 0.1
        fin_taper_ratio = 0.8
        fin_x_end = geo['l'] - 0.025 
        fin_x_start = fin_x_end - fin_length
        
        x_norm_fin_start = (fin_x_start - geo['lf']) / c
        r_fin_start = geo['c_offset'] + (r_max - geo['c_offset']) * (1 - x_norm_fin_start**geo['n'])
        x_norm_fin_end = (fin_x_end - geo['lf']) / c
        r_fin_end = geo['c_offset'] + (r_max - geo['c_offset']) * (1 - x_norm_fin_end**geo['n'])
        
        fin_verts = []
        v1 = [fin_x_start, r_fin_start, 0]; v2 = [fin_x_start, r_fin_start + fin_span, 0]
        v3 = [fin_x_end, r_fin_end + fin_span * fin_taper_ratio, 0]; v4 = [fin_x_end, r_fin_end, 0]
        fin_verts.append([v1, v2, v3, v4])
        v1 = [fin_x_start, -r_fin_start, 0]; v2 = [fin_x_start, -(r_fin_start + fin_span), 0]
        v3 = [fin_x_end, -(r_fin_end + fin_span * fin_taper_ratio), 0]; v4 = [fin_x_end, -r_fin_end, 0]
        fin_verts.append([v1, v2, v3, v4])
        v1 = [fin_x_start, 0, r_fin_start]; v2 = [fin_x_start, 0, r_fin_start + fin_span]
        v3 = [fin_x_end, 0, r_fin_end + fin_span * fin_taper_ratio]; v4 = [fin_x_end, 0, r_fin_end]
        fin_verts.append([v1, v2, v3, v4])
        v1 = [fin_x_start, 0, -r_fin_start]; v2 = [fin_x_start, 0, -(r_fin_start + fin_span)]
        v3 = [fin_x_end, 0, -(r_fin_end + fin_span * fin_taper_ratio)]; v4 = [fin_x_end, 0, -r_fin_end]
        fin_verts.append([v1, v2, v3, v4])
        
        # 4-Bladed Propeller
        prop_tip_radius = fin_span * 1.0 
        prop_hub_radius = r_final
        prop_pitch = 0.1
        prop_chord_angle = np.pi / 8
        prop_x_pos = geo['l']
        
        prop_blades = []
        r_blade = np.linspace(prop_hub_radius, prop_tip_radius, 8)
        th_blade_base = np.linspace(-prop_chord_angle/2, prop_chord_angle/2, 5)
        
        for i in range(4):
            base_angle = i * (np.pi / 2)
            R_blade, TH_blade = np.meshgrid(r_blade, th_blade_base)
            Y_blade = R_blade * np.cos(TH_blade + base_angle)
            Z_blade = R_blade * np.sin(TH_blade + base_angle)
            X_blade = prop_x_pos + (R_blade * prop_pitch) * np.sin(TH_blade)
            prop_blades.append((X_blade, Y_blade, Z_blade))

        # Protective Cage (Shroud)
        cage_radius = prop_tip_radius + 0.015
        cage_length = 0.08
        cage_x_start = geo['l'] - 0.02
        cage_x_end = cage_x_start + cage_length
        
        x_cage = np.linspace(cage_x_start, cage_x_end, 10)
        theta_cage = np.linspace(0, 2 * np.pi, 40)
        
        X_cage, TH_cage = np.meshgrid(x_cage, theta_cage)
        Y_cage = cage_radius * np.cos(TH_cage)
        Z_cage = cage_radius * np.sin(TH_cage)
        
        # Store all geometry
        self.base_geometry = {
            'nose': (X_nose, Y_nose, Z_nose),
            'mid': (X_mid, Y_mid, Z_mid),
            'tail': (X_tail, Y_tail, Z_tail),
            'sss1': (X_sss1, Y_sss1, Z_sss1),
            'sss2': (X_sss2, Y_sss2, Z_sss2),
            'dvl_faces': dvl_faces,
            'mast': (X_mast, Y_mast, Z_mast),
            'cage': (X_cage, Y_cage, Z_cage),
            'prop_blades': prop_blades,
        }
        self.fins = fin_verts

    def rotation_matrix(self, roll, pitch, yaw):
        """Create rotation matrix from Euler angles (ZYX convention)"""

        # Rotation around x-axis (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # Rotation around y-axis (pitch)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Rotation around z-axis (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation: Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def transform_geometry(self, X, Y, Z):
        """Apply current position and orientation to geometry"""

        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
        R = self.rotation_matrix(*self.orientation)
        rotated = R @ points
        translated = rotated + self.position.reshape(3, 1)

        X_new = translated[0].reshape(X.shape)
        Y_new = translated[1].reshape(Y.shape)
        Z_new = translated[2].reshape(Z.shape)

        return X_new, Y_new, Z_new
    
    def transform_fins(self):
        """Transform fin vertices"""
        transformed_fins = []
        R = self.rotation_matrix(*self.orientation)
        
        for fin in self.fins:
            fin_array = np.array(fin)
            # Apply rotation and translation
            rotated = (R @ fin_array.T).T
            translated = rotated + self.position
            transformed_fins.append(translated)
        
        return transformed_fins
    
    def transform_dvl(self):
        """Transform DVL box"""
        R = self.rotation_matrix(*self.orientation)
        transformed_faces = []
        
        for face in self.base_geometry['dvl_faces']:
            face_array = np.array(face)
            rotated = (R @ face_array.T).T
            translated = rotated + self.position
            transformed_faces.append(translated)
        
        return transformed_faces    
    
    def update_state(self):
        """Update the state of the AUV by keyboard inputs"""

        # Reset the position
        if 'r' in self.keys_pressed:
            self.position = np.array([0.0, 0.0, 0.0])
            self.orientation = np.array([0.0, 0.0, 0.0])
            self.velocity = 0.0
            return
        
        # Throttle input
        throttle = 0.0
        if 'w' in self.keys_pressed:
            throttle = 1.0
        elif 'x' in self.keys_pressed:
            throttle = -1.0

        # Update velocity with acceleratoin or deceleration

        # Brake
        if 'b' in self.keys_pressed:
            if self.velocity > 0:
                self.velocity -= self.deceleration * 2.0* self.dt
                self.velocity = max(0, self.velocity)
            elif self.velocity < 0:
                self.velocity += self.deceleration * 2.0 * self.dt 
                self.velocity = min(0, self.velocity)

        # Accelerate or decelerate based on throttle
        elif abs(throttle) > 0.01:
            target_velocity = throttle * self.max_velocity
            
            # Accelerate/Forward
            if target_velocity > self.velocity:
                self.velocity += self.acceleration * self.dt
                self.velocity = min(self.velocity, target_velocity, self.max_velocity)
            # Decelerate/Backward
            elif target_velocity < self.velocity:
                self.velocity -= self.acceleration * self.dt
                self.velocity = max(self.velocity, target_velocity, -self.max_velocity * 0.5)

        # Natural friction/drag
        else:
            if self.velocity > 0:
                self.velocity -= self.friction * self.dt
                self.velocity = max(0, self.velocity)
            elif self.velocity < 0:
                self.velocity += self.friction * self.dt
                self.velocity = min(0, self.velocity)

        






if __name__ == '__main__':
    
    old_L = 1.33
    old_a = 0.191        # Nose Length (m)
    old_a_offset = 0.0165 # Nose Offset (m)
    old_c_offset = 0.0368 # Tail Offset (m)
    old_lf = 0.828       # Vehicle Forward Length (m)

    new_L = REMUS_PARAMS["L"]  # 1.6 m
    new_D = REMUS_PARAMS["D"]  # 0.19 m
    
    # We pro-rate the visual section lengths to fit the new total length
    scale_ratio = new_L / old_L  # (1.6 / 1.33)
    
    auv_geo = {
        'a': old_a * scale_ratio,
        'a_offset': old_a_offset * scale_ratio,
        'c_offset': old_c_offset * scale_ratio,
        'n': 2,            # Exponential Coefficient (shape, does not scale)
        'd': new_D,
        'lf': old_lf * scale_ratio,
        'l': new_L,
    }
    
    # Calculate 'c' (tail length)
    calculated_c = auv_geo['l'] - auv_geo['lf'] 

    # Get physics marker positions
    cb_pos = PARAMS_DERIVED["cb_pos"]
    cg_pos = PARAMS_DERIVED["cg_pos"]
