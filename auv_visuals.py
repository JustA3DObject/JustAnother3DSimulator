import numpy as np
from auv_parameters import REMUS_PARAMS, PARAMS_DERIVED
from auv_physics import jacobian 

class AUVGeometry:
    """
    Handles the generation and transformation of the AUV's 3D mesh.
    
    """
    def __init__(self):
        # Load geometric params and origin
        self.geo = self._get_geometry_params()
        self.origin_vec_from_nose = np.array(PARAMS_DERIVED["cb_pos"])
        
        # Base geometry is stored here, centered at the origin (CB)
        self.base_geometry = {}
        # self.fins is a list of fins, where each fin is a list of faces (polygons)
        self.fins = []
        
        # Generate the geometry on initialization
        self._generate_base_geometry()

    def _get_geometry_params(self):
        """Helper to scale AUV geometry based on parameters."""
        old_L = 1.33
        old_a = 0.191
        old_a_offset = 0.0165
        old_c_offset = 0.0368
        old_lf = 0.828
        
        new_L = REMUS_PARAMS["L"]
        new_D = REMUS_PARAMS["D"]
        scale_ratio = new_L / old_L
        
        return {
            'a': old_a * scale_ratio,
            'a_offset': old_a_offset * scale_ratio,
            'c_offset': old_c_offset * scale_ratio,
            'n': 2,
            'd': new_D,
            'lf': old_lf * scale_ratio,
            'l': new_L,
        }

    def _rotation_matrix(self, roll, pitch, yaw):
        """ Create rotation matrix from Euler angles (ZYX convention)"""
        temp_eta = np.array([[0],[0],[0],[roll],[pitch],[yaw]])
        R = jacobian(temp_eta)[0:3, 0:3]
        return R

    def _transform_points(self, X, Y, Z, R, position):
        """Apply rotation and translation to a set of points (Surface/MeshGrid style)."""
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
        rotated = R @ points
        translated = rotated + position.reshape(3, 1)

        X_new = translated[0].reshape(X.shape)
        Y_new = translated[1].reshape(Y.shape)
        Z_new = translated[2].reshape(Z.shape)
        return X_new, Y_new, Z_new

    def get_transformed_assets(self, position, orientation):
        """
        Transforms all base geometry to the current AUV position/orientation.
        Returns a dictionary of assets ready for plotting.
        """
        R = self._rotation_matrix(*orientation)
        assets = {}

        # Transform hull sections
        for part in ['nose', 'mid', 'tail', 'sss1', 'sss2', 'mast', 'cage']:
            X, Y, Z = self.base_geometry[part]
            assets[part] = self._transform_points(X, Y, Z, R, position)
            
        # Transform propeller blades
        assets['prop_blades'] = []
        for X_b, Y_b, Z_b in self.base_geometry['prop_blades']:
            assets['prop_blades'].append(
                self._transform_points(X_b, Y_b, Z_b, R, position)
            )

        # Transform DVL faces
        assets['dvl_faces'] = []
        for face in self.base_geometry['dvl_faces']:
            face_array = np.array(face)
            rotated = (R @ face_array.T).T
            translated = rotated + position
            assets['dvl_faces'].append(translated)
            
        # Transform 3D Fins
        # Structure: List of Fins -> List of Faces -> Array of Vertices
        assets['fins'] = []
        for fin_faces in self.fins:
            transformed_fin = []
            for face in fin_faces:
                # 'face' is an (N, 3) numpy array
                rotated = (R @ face.T).T
                translated = rotated + position
                transformed_fin.append(translated)
            assets['fins'].append(transformed_fin)
            
        return assets

    def _generate_prism_fin(self, root_x, root_z, tip_x, tip_z, thickness):
        """
        Generates a triangular prism fin.
        Adapted from generate_meshes.py to work with numpy arrays directly.
        
        Returns a list of faces, where each face is an (N, 3) numpy array.
        """
        t_half = thickness / 2
        
        # 6 Vertices for a triangular prism
        # Root Leading Edge (Center - Pointed)
        # Tip Leading Edge (Center - Pointed)
        # Root Trailing Edge (Top)
        # Root Trailing Edge (Bottom)
        # Tip Trailing Edge (Top)
        # Tip Trailing Edge (Bottom)
        
        verts = np.array([
            [root_x[0], 0.0, root_z[0]],       
            [tip_x[0],  0.0, tip_z[0]],        
            [root_x[1], t_half, root_z[1]],    
            [root_x[1], -t_half, root_z[1]],   
            [tip_x[1],  t_half, tip_z[1]],     
            [tip_x[1],  -t_half, tip_z[1]]     
        ])
        
        # Define Faces (indices)
        # Ensure CCW winding order for outside normals
        face_indices = [
            [0, 2, 4, 1], # Top Slope
            [0, 1, 5, 3], # Bottom Slope
            [2, 3, 5, 4], # Back Face
            [0, 3, 2],    # Root Side (Triangle)
            [1, 4, 5]     # Tip Side (Triangle)
        ]
        
        faces = []
        for idx_list in face_indices:
            faces.append(verts[idx_list])
            
        return faces

    def _generate_base_geometry(self):
        """
        Generate the AUV geometry.
        All geometry is first created relative to the nose (x=0),
        then shifted so the origin (0,0,0) is at the Center of Buoyancy.
        """
        geo = self.geo
        
        r_max = geo['d'] / 2
        num_x_points = 50 
        num_theta_points = 40
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
        sss_len = 0.3; sss_width_angle = 0.1
        x_sss_start = geo['a'] + (mid_section_length - sss_len) / 2
        x_sss_end = x_sss_start + sss_len
        x_sss = np.linspace(x_sss_start, x_sss_end, 10)
        th_sss1 = np.linspace(np.pi - sss_width_angle, np.pi + sss_width_angle, 10)
        X_sss1, TH_sss1 = np.meshgrid(x_sss, th_sss1); R_sss1 = r_max + z_offset
        Y_sss1 = R_sss1 * np.cos(TH_sss1); Z_sss1 = R_sss1 * np.sin(TH_sss1)
        th_sss2 = np.linspace(-sss_width_angle, sss_width_angle, 10)
        X_sss2, TH_sss2 = np.meshgrid(x_sss, th_sss2); R_sss2 = r_max + z_offset
        Y_sss2 = R_sss2 * np.cos(TH_sss2); Z_sss2 = R_sss2 * np.sin(TH_sss2)

        # DVL (Doppler Velocity Log) Box
        dvl_len = 0.1; dvl_width = 0.08; dvl_height = 0.03
        x_dvl_start = geo['lf'] - dvl_len - 0.05
        x_dvl_end = x_dvl_start + dvl_len
        y_dvl_half = dvl_width / 2
        z_dvl_top = -r_max; z_dvl_bottom = z_dvl_top - dvl_height
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
        mast_height = 0.08; mast_radius = 0.01
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
        fin_thickness = 0.02
        
        fin_x_end = geo['l'] - 0.025
        fin_x_start = fin_x_end - fin_length
        
        # Calculate hull radius at fin start and end positions
        x_norm_fin_start = (fin_x_start - geo['lf']) / c
        r_fin_start = geo['c_offset'] + (r_max - geo['c_offset']) * (1 - x_norm_fin_start**geo['n'])
        
        x_norm_fin_end = (fin_x_end - geo['lf']) / c
        r_fin_end = geo['c_offset'] + (r_max - geo['c_offset']) * (1 - x_norm_fin_end**geo['n'])
        
        self.fins = [] # Reset fins list
        
        # Generate 4 fins (0, 90, 180, 270 degrees)
        for i in range(4):
            angle = i * (np.pi / 2)
            
            # Generate local 3D fin faces (prism)
            # The _generate_prism_fin function assumes the fin grows in +Z direction (vertical)
            # We map the radius (r) to the Z coordinate
            local_faces = self._generate_prism_fin(
                root_x=[fin_x_start, fin_x_end],
                root_z=[r_fin_start - 0.005, r_fin_end - 0.005], # Embed slightly in hull
                tip_x=[fin_x_start + 0.02, fin_x_end],
                tip_z=[r_fin_start + fin_span, r_fin_end + fin_span * fin_taper_ratio],
                thickness=fin_thickness
            )
            
            # Rotate fin to its correct position (0, 90, 180, 270)
            # This rotation is about the X-axis (Roll)
            R_fin = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
            
            rotated_faces = []
            for face in local_faces:
                # Rotate the face vertices
                # face is (N, 3), so we do (R @ face.T).T
                rotated_face = (R_fin @ face.T).T
                rotated_faces.append(rotated_face)
                
            self.fins.append(rotated_faces)

        # 4-Bladed Propeller
        prop_tip_radius = fin_span * 1.0; prop_hub_radius = r_final
        prop_pitch = 0.1; prop_chord_angle = np.pi / 8; prop_x_pos = geo['l']
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
        cage_radius = prop_tip_radius + 0.015; cage_length = 0.08
        cage_x_start = geo['l'] - 0.02; cage_x_end = cage_x_start + cage_length
        x_cage = np.linspace(cage_x_start, cage_x_end, 10)
        theta_cage = np.linspace(0, 2 * np.pi, 40)
        X_cage, TH_cage = np.meshgrid(x_cage, theta_cage)
        Y_cage = cage_radius * np.cos(TH_cage); Z_cage = cage_radius * np.sin(TH_cage)
        
        # Re-center All Geometry around CB
        origin_x, origin_y, origin_z = self.origin_vec_from_nose

        # Re-center hull sections
        X_nose -= origin_x; Y_nose -= origin_y; Z_nose -= origin_z
        X_mid  -= origin_x; Y_mid  -= origin_y; Z_mid  -= origin_z
        X_tail -= origin_x; Y_tail -= origin_y; Z_tail -= origin_z
        
        # Re-center SSS patches
        X_sss1 -= origin_x; Y_sss1 -= origin_y; Z_sss1 -= origin_z
        X_sss2 -= origin_x; Y_sss2 -= origin_y; Z_sss2 -= origin_z
        
        # Re-center DVL faces
        v_recentered = v - self.origin_vec_from_nose
        dvl_faces_recentered = [
            [v_recentered[0], v_recentered[1], v_recentered[2], v_recentered[3]], 
            [v_recentered[4], v_recentered[5], v_recentered[6], v_recentered[7]], 
            [v_recentered[0], v_recentered[1], v_recentered[5], v_recentered[4]],
            [v_recentered[2], v_recentered[3], v_recentered[7], v_recentered[6]], 
            [v_recentered[0], v_recentered[3], v_recentered[7], v_recentered[4]], 
            [v_recentered[1], v_recentered[2], v_recentered[6], v_recentered[5]]
        ]

        # Re-center Mast
        X_mast -= origin_x; Y_mast -= origin_y; Z_mast -= origin_z

        # Re-center Propeller Blades
        prop_blades_recentered = []
        for (X_b, Y_b, Z_b) in prop_blades:
            prop_blades_recentered.append((
                X_b - origin_x, Y_b - origin_y, Z_b - origin_z
            ))

        # Re-center Cage
        X_cage -= origin_x; Y_cage -= origin_y; Z_cage -= origin_z

        # Re-center Fins
        fin_faces_recentered = []
        for fin_faces in self.fins:
            # fin_faces is a list of (N,3) arrays
            new_fin = []
            for face in fin_faces:
                new_fin.append(face - self.origin_vec_from_nose)
            fin_faces_recentered.append(new_fin)
        self.fins = fin_faces_recentered

        # Store Geometry
        self.base_geometry = {
            'nose': (X_nose, Y_nose, Z_nose),
            'mid': (X_mid, Y_mid, Z_mid),
            'tail': (X_tail, Y_tail, Z_tail),
            'sss1': (X_sss1, Y_sss1, Z_sss1),
            'sss2': (X_sss2, Y_sss2, Z_sss2),
            'dvl_faces': dvl_faces_recentered,
            'mast': (X_mast, Y_mast, Z_mast),
            'cage': (X_cage, Y_cage, Z_cage),
            'prop_blades': prop_blades_recentered,
        }