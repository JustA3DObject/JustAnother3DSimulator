import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from auv_parameters import REMUS_PARAMS, PARAMS_DERIVED
from auv_hydrodynamic_parameters import HYDRO_PARAMS
from matplotlib.animation import FuncAnimation

# Helper Functions 


def skew(v):
    """ Converts a 3-element vector to a 3x3 skew-symmetric matrix"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def jacobian(eta):
    """ Computes the 6-DOF Jacobian matrix J(eta)"""
    phi, theta, psi = eta[3:6].flatten()
    
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    
    # J1 (Linear) - Fossen/SNAME convention
    J1 = np.array([
        [cpsi*cth, -spsi*cphi + cpsi*sth*sphi, spsi*sphi + cpsi*sth*cphi],
        [spsi*cth, cpsi*cphi + spsi*sth*sphi, -cpsi*sphi + spsi*sth*cphi],
        [-sth, cth*sphi, cth*cphi]
    ])
    # J2 (Angular)
    J2 = np.array([
        [1, sphi*np.tan(theta), cphi*np.tan(theta)],
        [0, cphi, -sphi],
        [0, sphi/cth, cphi/cth]
    ])
    
    J = np.zeros((6, 6))
    J[0:3, 0:3] = J1
    J[3:6, 3:6] = J2
    return J

class AUVPhysicsModel:
    """ Implements the 6-DOF dynamic model for the AUV"""
    def __init__(self):
        # Load physical and hydrodynamic parameters
        self.params = {**REMUS_PARAMS, **HYDRO_PARAMS}
        
        # State Vectors
        # eta: [x, y, z, phi, theta, psi] (World-frame position and orientation)
        self.eta = np.zeros((6, 1))
        self.eta[2] = -5.0 # Start 5m deep
        # nu: [u, v, w, p, q, r] (Body-frame linear and angular velocities)
        self.nu = np.zeros((6, 1))
        
        # Physical Constants
        self.m = self.params['m']
        self.L = self.params['L']
        self.W = self.params['W']
        self.B = self.params['B']
        self.g = 9.81

        # Define body-frame vectors
        # Vector from CB (origin) to CB
        self.r_b = np.array([0., 0., 0.])
        
        # Calculate CB position relative to the nose
        r_b_vec_nose = np.array(PARAMS_DERIVED["cb_pos"])
        # Calculate CG position relative to the nose
        r_g_vec_nose = np.array(PARAMS_DERIVED["cg_pos"])
        
        # Vector from CB (origin) to CG
        self.r_g = (r_g_vec_nose - r_b_vec_nose).reshape(3, 1)
        
        # Inertia tensor
        self.I_o = np.diag([
            self.params['Ixx'],
            self.params['Iyy'],
            self.params['Izz']
        ])
        
        # System Matrices
        self.build_mass_matrices()
        self.build_damping_matrices()
        
        # Control Limits
        self.MAX_THRUST = 40.0 # Newtons (approx. 4kg-force)
        self.MAX_RUDDER_ANGLE = np.radians(20) # rad
        self.MAX_STERN_ANGLE = np.radians(20) # rad

    def build_mass_matrices(self):
        """ Builds the M_RB and M_A matrices based on CB origin"""
        m = self.m
        
        # Rigid-Body Mass Matrix (M_RB)
        # M_RB = [ m*I, -m*S(r_g)]
        #        [ m*S(r_g), I_o]
        S_rg = skew(self.r_g.flatten())
        
        self.M_RB = np.zeros((6, 6))
        self.M_RB[0:3, 0:3] = np.diag([m, m, m])
        self.M_RB[0:3, 3:6] = -m * S_rg
        self.M_RB[3:6, 0:3] = m * S_rg
        self.M_RB[3:6, 3:6] = self.I_o
        
        # Added Mass Matrix (M_A)
        self.M_A = np.diag([
            -self.params['Xu_dot'],
            -self.params['Yv_dot'],
            -self.params['Zw_dot'],
            -self.params['Kp_dot'],
            -self.params['Mq_dot'],
            -self.params['Nr_dot']
        ])
        # Off-diagonal added mass terms
        self.M_A[1, 5] = self.M_A[5, 1] = -self.params.get('Y_r_dot', 0)
        self.M_A[2, 4] = self.M_A[4, 2] = -self.params.get('Z_q_dot', 0)
        self.M_A[4, 2] = self.M_A[2, 4] = -self.params.get('M_w_dot', 0)
        self.M_A[5, 1] = self.M_A[1, 5] = -self.params.get('N_v_dot', 0)
        
        # Total Mass Matrix
        self.M = self.M_RB + self.M_A
        self.M_inv = inv(self.M)

    def build_damping_matrices(self):
        """ Builds the linear and quadratic damping matrices"""
        # Linear Damping (D_lin)
        self.D_lin = -np.diag([
            self.params['Xu'],
            self.params['Yv'],
            self.params['Zw'],
            self.params['Kp'],
            self.params['Mq'],
            self.params['Nr']
        ])
        
        # Quadratic Damping (Dn) - Store coefficients
        self.D_quad_coeffs = -np.diag([
            self.params['X_u|u|'],
            self.params['Y_v|v|'],
            self.params['Z_w|w|'],
            self.params['K_p|p|'],
            self.params['M_q|q|'],
            self.params['N_r|r|']
        ])

    def reset(self):
        """ Resets the AUV state."""
        self.eta = np.zeros((6, 1))
        self.eta[2] = -5.0 # Reset to 5m deep
        self.nu = np.zeros((6, 1))

    def calculate_hydrostatics(self, eta):
        """ Calculates g(eta) (restoring forces (gravity, buoyancy))"""
        phi, theta, psi = eta[3:6].flatten()
        
        # Rotation matrix from body to world
        R_b_w = jacobian(eta)[0:3, 0:3]
        
        # Gravity force (acts at CG (r_g))
        f_g_world = np.array([[0], [0], [self.W]])
        f_g_body = R_b_w.T @ f_g_world
        tau_g_body = skew(self.r_g.flatten()) @ f_g_body
        
        # Buoyancy force (acts at CB (origin, r_b))
        f_b_world = np.array([[0], [0], [-self.B]])
        f_b_body = R_b_w.T @ f_b_world
        tau_b_body = np.zeros((3, 1)) # Moment is 0 since r_b = [0,0,0]
        
        # Total restoring force (moment vector)
        g_eta = np.vstack((f_g_body + f_b_body, tau_g_body + tau_b_body))
        return g_eta

    def calculate_damping(self, nu):
        """ Calculates D(nu)*nu (linear and quadratic damping)"""
        D_nu = self.D_lin @ nu
        
        # Add quadratic damping
        D_nu_quad = self.D_quad_coeffs @ (np.abs(nu) * nu)
        
        return D_nu + D_nu_quad

    def calculate_coriolis(self, nu):
        """
        Calculates C(nu)*nu = C_RB(nu)*nu + C_A(nu)*nu
        C_RB is based on the Newton-Euler equations.
        C_A is a simplified matrix for added mass.
        """
        # Rigid-Body Coriolis C_RB(nu)*nu        
        u, v, w, p, q, r = nu.flatten()
        m = self.m
        xg, yg, zg = self.r_g.flatten()
        Ixx, Iyy, Izz = self.params['Ixx'], self.params['Iyy'], self.params['Izz']
        
        C_RB_nu = np.zeros((6, 1))
        
        # SURGE
        C_RB_nu[0] = m * (-v * r + w * q - xg * (q**2 + r**2) + yg * (p * q) + zg * (p * r))

        # SWAY
        C_RB_nu[1] = m * (-w * p + u * r - yg * (p**2 + r**2) + zg * (q * r) + xg * (p * q))

        # HEAVE
        C_RB_nu[2] = m * (-u * q + v * p - zg * (p**2 + q**2) + xg * (r * p) + yg * (r * q))

        # ROLL
        C_RB_nu[3] = (Izz - Iyy) * q * r + m * (yg * (-u * q + v * p) - zg * (-w * p + u * r))

        # PITCH
        C_RB_nu[4] = (Ixx - Izz) * p * r + m * (zg * (-v * r + w * q) - xg * (-u * q + v * p))

        # YAW
        C_RB_nu[5] = (Iyy - Ixx) * p * q + m * (xg * (-w * p + u * r) - yg * (-v * r + w * q))
        
        # Added Mass Coriolis C_A(nu)*nu
        # (Simplified, assuming diagonal M_A for C_A calculation)
        A11, A22, A33 = self.M_A[0,0], self.M_A[1,1], self.M_A[2,2]
        A44, A55, A66 = self.M_A[3,3], self.M_A[4,4], self.M_A[5,5]

        C_A_nu = np.zeros((6, 1))
        C_A_nu[0] = (A33*w*q - A22*v*r)
        C_A_nu[1] = (A11*u*r - A33*w*p)
        C_A_nu[2] = (A22*v*p - A11*u*q)
        C_A_nu[3] = (A66*r*q - A55*q*r)
        C_A_nu[4] = (A44*p*r - A66*r*p)
        C_A_nu[5] = (A55*q*p - A44*p*q)
        
        return C_RB_nu + C_A_nu


    def calculate_control_forces(self, nu, control_cmds):
        """ Calculates tau_control based on inputs"""
        # Use u (surge vel) for control effectiveness
        # Use abs(u) to handle reverse
        u = nu[0, 0] 
        u_eff = abs(u)
        
        # Get commands
        throttle_cmd = control_cmds['throttle'] # 1.0 (fwd), -0.5 (rev)
        yaw_cmd = control_cmds['yaw'] # 1.0 (left), -1.0 (right)
        pitch_cmd = control_cmds['pitch'] # 1.0 (down), -1.0 (up)

        # Thrust
        X_thrust = throttle_cmd * self.MAX_THRUST
        
        # Rudder (Yaw)
        delta_r = yaw_cmd * self.MAX_RUDDER_ANGLE
        # Y = Yuu_dr * |u|*u * dr  (Using |u|*u for stability)
        Y_rudder = self.params['Yuu_delta_r'] * (u_eff * u) * delta_r
        # N = Nuu_dr * |u|*u * dr
        N_rudder = self.params['Nuu_delta_r'] * (u_eff * u) * delta_r
        
        # Stern Plane (Pitch)
        delta_s = pitch_cmd * self.MAX_STERN_ANGLE
        # Z = Zuu_ds * |u|*u * ds
        Z_stern = self.params['Zuu_delta_s'] * (u_eff * u) * delta_s
        # M = Muu_ds * |u|*u * ds
        M_stern = self.params['Muu_delta_s'] * (u_eff * u) * delta_s
        
        # Roll (K) - not controlled
        K_control = 0.0
        
        tau_control = np.array([
            [X_thrust],
            [Y_rudder],
            [Z_stern],
            [K_control],
            [M_stern],
            [N_rudder]
        ])
        return tau_control

    def step(self, dt, control_cmds):
        """
        Advances the simulation by one time step dt.
        M*nu_dot + C(nu)*nu + D(nu)*nu + g(eta) = tau
        """
        # Calculate all forces and moments in the body frame
        tau_G = self.calculate_hydrostatics(self.eta)
        tau_D = self.calculate_damping(self.nu)
        tau_C = self.calculate_coriolis(self.nu)
        tau_ctrl = self.calculate_control_forces(self.nu, control_cmds)
        
        # Sum all forces
        # We use -g, -D, -C because they are on the LHS of the equation
        tau_total = tau_ctrl - tau_G - tau_D - tau_C
        
        # Solve for acceleration: nu_dot = M_inv * tau_total
        nu_dot = self.M_inv @ tau_total
        
        # Integrate acceleration (Euler step)
        self.nu = self.nu + nu_dot * dt
        
        # Transform body velocities to world-frame velocities
        # Handle gimbal lock at +/- 90 deg pitch
        theta = self.eta[4, 0]
        if abs(np.cos(theta)) < 1e-6:
             self.eta[4, 0] = np.sign(theta) * (np.pi/2 - 1e-3) # nudge
        
        J = jacobian(self.eta)
        eta_dot = J @ self.nu
        
        # Integrate position
        self.eta = self.eta + eta_dot * dt
        
        # Surface Interaction
        # Don't let AUV go above the water (Z=0)
        if self.eta[2, 0] > 0:
            self.eta[2, 0] = 0
            # Stop any upward body-frame velocity (w)
            if self.nu[2, 0] < 0: # w is positive down, so check for negative w
                self.nu[2, 0] = 0
            # Also stop any upward world-frame velocity
            if eta_dot[2, 0] > 0:
                self.nu = inv(J) @ np.array([eta_dot[0,0], eta_dot[1,0], 0, eta_dot[3,0], eta_dot[4,0], eta_dot[5,0]]).reshape(6,1)

# Controller 

class AUVController: 
    """ Controller/Visualizer to make the AUV interactive"""
    def __init__(self, geometry):
        # Store geometry parameters
        self.geometry = geometry

        # Create the physics model
        self.physics = AUVPhysicsModel()

        # Get the new origin (CB) position relative to the nose
        # This is used ONLY for plotting the geometry, not for physics
        self.origin_vec_from_nose = np.array(PARAMS_DERIVED["cb_pos"])

        # Read state from physics model
        self.position = self.physics.eta[0:3].flatten()
        self.orientation = self.physics.eta[3:6].flatten()
        self.velocity = self.physics.nu[0, 0]

        # Movement parameters
        self.max_velocity = 2.0 # For display only
        self.dt = 0.05 # s

        # Key tracking
        self.keys_pressed = set()

        # Generating geometry
        self.generate_base_geometry()

    def generate_base_geometry(self):
        """
        Generate the AUV geometry.
        All geometry is first created relative to the nose (x=0),
        then shifted so the origin (0,0,0) is at the Center of Buoyancy.
        """
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
        fin_verts_recentered = []
        for fin in fin_verts:
            fin_array_recentered = np.array(fin) - self.origin_vec_from_nose
            fin_verts_recentered.append(fin_array_recentered)

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
        self.fins = fin_verts_recentered

    def rotation_matrix(self, roll, pitch, yaw):
        """ Create rotation matrix from Euler angles (ZYX convention)"""
        temp_eta = np.array([[0],[0],[0],[roll],[pitch],[yaw]])
        R = jacobian(temp_eta)[0:3, 0:3]
        return R
    
    def transform_geometry(self, X, Y, Z):
        """ Apply current position and orientation to geometry"""
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
        R = self.rotation_matrix(*self.orientation)
        rotated = R @ points
        translated = rotated + self.position.reshape(3, 1)

        X_new = translated[0].reshape(X.shape)
        Y_new = translated[1].reshape(Y.shape)
        Z_new = translated[2].reshape(Z.shape)
        return X_new, Y_new, Z_new
    
    def transform_fins(self):
        """ Transform fin vertices"""
        transformed_fins = []
        R = self.rotation_matrix(*self.orientation)
        for fin_array in self.fins:
            rotated = (R @ fin_array.T).T
            translated = rotated + self.position
            transformed_fins.append(translated)
        return transformed_fins
    
    def transform_dvl(self):
        """ Transform DVL box"""
        R = self.rotation_matrix(*self.orientation)
        transformed_faces = []
        for face in self.base_geometry['dvl_faces']:
            face_array = np.array(face)
            rotated = (R @ face_array.T).T
            translated = rotated + self.position
            transformed_faces.append(translated)
        return transformed_faces    
    
    def update_state(self):
        """
        Update the state by sending commands to the physics model
        and then reading its new state.
        """
        
        # Parse keyboard inputs into control commands
        control_cmds = {
            'throttle': 0.0,
            'yaw': 0.0,
            'pitch': 0.0
        }
        
        if 'r' in self.keys_pressed:
            self.physics.reset()

        # 'W' = Forward = Positive X-direction = Positive Thrust
        if 'w' in self.keys_pressed:
            control_cmds['throttle'] = 1.0  # Full forward thrust
        # 'X' = Backward = Negative X-direction = Negative Thrust
        elif 'x' in self.keys_pressed:
            control_cmds['throttle'] = -0.5 # Half reverse thrust
            
        # Brake (applies counter-thrust)
        if 'b' in self.keys_pressed:
            if self.physics.nu[0,0] > 0.1: # If moving forward (u > 0)
                control_cmds['throttle'] = -1.0 # Apply reverse (negative)
            elif self.physics.nu[0,0] < -0.1: # If moving backward (u < 0)
                control_cmds['throttle'] = 1.0 # Apply forward (positive)

        # Yaw (Rudder)
        if 'a' in self.keys_pressed:
            control_cmds['yaw'] = 1.0  # Rudder left
        if 'd' in self.keys_pressed:
            control_cmds['yaw'] = -1.0 # Rudder right
        
        # Pitch (Stern planes)
        if 'z' in self.keys_pressed:
            control_cmds['pitch'] = -1.0 # Sterns up -> Nose up
        if 'c' in self.keys_pressed:
            control_cmds['pitch'] = 1.0  # Sterns down -> Nose down
        
        # Step the physics model
        self.physics.step(self.dt, control_cmds)
        
        # Read the new state back from the physics model
        self.position = self.physics.eta[0:3].flatten()
        self.orientation = self.physics.eta[3:6].flatten()
        self.velocity = self.physics.nu[0, 0] # Read surge velocity

        # Normalize angles for display/rotation
        self.orientation[0] = np.arctan2(np.sin(self.orientation[0]), np.cos(self.orientation[0]))
        self.orientation[1] = np.arctan2(np.sin(self.orientation[1]), np.cos(self.orientation[1]))
        self.orientation[2] = np.arctan2(np.sin(self.orientation[2]), np.cos(self.orientation[2]))
    
# Main Visualization
        
def create_interactive_auv():
    """Create interactive AUV with keyboard controls"""
    
    # Setup geometry
    old_L = 1.33
    old_a = 0.191
    old_a_offset = 0.0165
    old_c_offset = 0.0368
    old_lf = 0.828
    
    new_L = REMUS_PARAMS["L"]
    new_D = REMUS_PARAMS["D"]
    
    scale_ratio = new_L / old_L
    
    auv_geo = {
        'a': old_a * scale_ratio,
        'a_offset': old_a_offset * scale_ratio,
        'c_offset': old_c_offset * scale_ratio,
        'n': 2,
        'd': new_D,
        'lf': old_lf * scale_ratio,
        'l': new_L,
    }
    
    # Create controller (which now creates the physics model)
    controller = AUVController(auv_geo)
    
    # Setup figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Store surface objects
    surf_nose = None
    surf_mid = None
    surf_tail = None
    surf_sss1 = None
    surf_sss2 = None
    surf_mast = None
    surf_cage = None
    dvl_collection = None
    fin_collections = []
    prop_surfaces = []
    
    # Control instructions
    instructions = (
        "--- 6-DOF DYNAMIC SIMULATOR\n"
        "W - Throttle Forward\n"
        "X - Throttle Backward\n"
        "A - Yaw Left | D - Yaw Right\n"
        "Z - Pitch Up | C - Pitch Down\n"
        "B - Emergency Brake\n"
        "R - Reset\n"
        "AUV is positively buoyant and hydrostatically stable."
    )
    
    # Add text for instructions
    fig.text(0.02, 0.98, instructions, transform=fig.transFigure,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # State display
    state_text = ax.text2D(0.02, 0.02, '', transform=ax.transAxes,
                           fontsize=9, family='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def on_key_press(event):
        """Handle key press events"""
        if event.key in ['w', 'x', 'a', 'd', 'z', 'c', 'b', 'r']:
            controller.keys_pressed.add(event.key)
    
    def on_key_release(event):
        """Handle key release events"""
        if event.key in controller.keys_pressed:
            controller.keys_pressed.discard(event.key)
    
    def init():
        """Initialize animation"""
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title('AUV 6-DOF Dynamic Simulator (CB-Centered)', fontsize=12)
        
        limit = 3
        pos = controller.position
        ax.set_xlim(pos[0] - limit, pos[0] + limit)
        ax.set_ylim(pos[1] - limit, pos[1] + limit)
        ax.set_zlim(pos[2] - limit, pos[2] + limit)
        
        return []
    
    def update(frame):
        """Animation update function"""
        nonlocal surf_nose, surf_mid, surf_tail, fin_collections
        nonlocal surf_sss1, surf_sss2, surf_mast, surf_cage, dvl_collection, prop_surfaces
        
        # Update state based on keyboard
        controller.update_state()
        
        # Clear previous artists
        if surf_nose: surf_nose.remove()
        if surf_mid: surf_mid.remove()
        if surf_tail: surf_tail.remove()
        if surf_sss1: surf_sss1.remove()
        if surf_sss2: surf_sss2.remove()
        if surf_mast: surf_mast.remove()
        if surf_cage: surf_cage.remove()
        if dvl_collection: dvl_collection.remove()
        for fc in fin_collections: fc.remove()
        fin_collections.clear()
        for ps in prop_surfaces: ps.remove()
        prop_surfaces.clear()
        
        # Transform and plot main hull geometry
        X_nose, Y_nose, Z_nose = controller.transform_geometry(
            *controller.base_geometry['nose'])
        X_mid, Y_mid, Z_mid = controller.transform_geometry(
            *controller.base_geometry['mid'])
        X_tail, Y_tail, Z_tail = controller.transform_geometry(
            *controller.base_geometry['tail'])
        
        surf_nose = ax.plot_surface(X_nose, Y_nose, Z_nose, 
                                    color='blue', alpha=0.7, rstride=5, cstride=5)
        surf_mid = ax.plot_surface(X_mid, Y_mid, Z_mid, 
                                   color='green', alpha=0.7, rstride=5, cstride=5)
        surf_tail = ax.plot_surface(X_tail, Y_tail, Z_tail, 
                                    color='red', alpha=0.7, rstride=5, cstride=5)
        
        # Plot other parts
        X_sss1, Y_sss1, Z_sss1 = controller.transform_geometry(*controller.base_geometry['sss1'])
        X_sss2, Y_sss2, Z_sss2 = controller.transform_geometry(*controller.base_geometry['sss2'])
        surf_sss1 = ax.plot_surface(X_sss1, Y_sss1, Z_sss1, color='grey', alpha=0.8)
        surf_sss2 = ax.plot_surface(X_sss2, Y_sss2, Z_sss2, color='grey', alpha=0.8)
        
        dvl_faces_transformed = controller.transform_dvl()
        dvl_collection = Poly3DCollection(dvl_faces_transformed, facecolors='orange', alpha=0.8)
        ax.add_collection3d(dvl_collection)
        
        X_mast, Y_mast, Z_mast = controller.transform_geometry(*controller.base_geometry['mast'])
        surf_mast = ax.plot_surface(X_mast, Y_mast, Z_mast, color='silver', alpha=0.9)
        
        X_cage, Y_cage, Z_cage = controller.transform_geometry(*controller.base_geometry['cage'])
        surf_cage = ax.plot_surface(X_cage, Y_cage, Z_cage, color='grey', alpha=0.4)
        
        transformed_fins = controller.transform_fins()
        for fin_verts in transformed_fins:
            fin_col = Poly3DCollection([fin_verts], facecolors='darkslategrey', edgecolors='black', alpha=0.9)
            ax.add_collection3d(fin_col)
            fin_collections.append(fin_col)
        
        for X_b, Y_b, Z_b in controller.base_geometry['prop_blades']:
            X_prop, Y_prop, Z_prop = controller.transform_geometry(X_b, Y_b, Z_b)
            prop_surf = ax.plot_surface(X_prop, Y_prop, Z_prop, color='black', alpha=0.9)
            prop_surfaces.append(prop_surf)
        
        # Update camera to follow AUV's CB
        pos = controller.position
        offset = 4
        ax.set_xlim(pos[0] - offset, pos[0] + offset)
        ax.set_ylim(pos[1] - offset, pos[1] + offset)
        ax.set_zlim(pos[2] - offset, pos[2] + offset)
        
        # Update state text
        pos = controller.physics.eta[0:3].flatten()
        vel = controller.physics.nu.flatten()
        ori = controller.physics.eta[3:6].flatten()
        
        # Display surge velocity (u) as "Speed"
        # Since u is negative for forward, we take abs(vel[0])
        speed = abs(vel[0]) 
        
        state_text.set_text(
            f"Pos (x,y,z): {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}\n"
            f"Vel (u,v,w): {vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f} m/s\n"
            f"Ori (r,p,y): {np.degrees(ori[0]):.1f}, {np.degrees(ori[1]):.1f}, {np.degrees(ori[2]):.1f}°\n"
            f"AngVel (p,q,r): {np.degrees(vel[3]):.1f}, {np.degrees(vel[4]):.1f}, {np.degrees(vel[5]):.1f}°/s"
        )
        
        return ([surf_nose, surf_mid, surf_tail, surf_sss1, surf_sss2, 
                surf_mast, surf_cage, dvl_collection] + 
                fin_collections + prop_surfaces)
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    
    # Create animation
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=None, interval=controller.dt * 1000, blit=False)
    
    plt.show()
    

if __name__ == '__main__':
    print("AUV 6-DOF Dynamic Simulator")
    print("\nControls:")
    print("  W - Accelerate Forward")
    print("  X - Accelerate Backward")
    print("  A - Yaw Left (Turn Left)")
    print("  D - Yaw Right (Turn Right)")
    print("  Z - Pitch Up (Nose Up)")
    print("  C - Pitch Down (Nose Down)")
    print("  B - Brake")
    print("  R - Reset")
    create_interactive_auv()