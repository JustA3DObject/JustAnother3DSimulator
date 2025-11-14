import numpy as np
from numpy.linalg import inv
from auv_parameters import REMUS_PARAMS, PARAMS_DERIVED
from auv_hydrodynamic_parameters import HYDRO_PARAMS

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

# Physics Model Class

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
        """
        # Rigid-Body Coriolis C_RB(nu)*nu        
        u, v, w, p, q, r = nu.flatten()
        m = self.m
        xg, yg, zg = self.r_g.flatten()
        Ixx, Iyy, Izz = self.params['Ixx'], self.params['Iyy'], self.params['Izz']
        
        C_RB_nu = np.zeros((6, 1))
        C_RB_nu[0] = m * (-v * r + w * q - xg * (q**2 + r**2) + yg * (p * q) + zg * (p * r))
        C_RB_nu[1] = m * (-w * p + u * r - yg * (p**2 + r**2) + zg * (q * r) + xg * (p * q))
        C_RB_nu[2] = m * (-u * q + v * p - zg * (p**2 + q**2) + xg * (r * p) + yg * (r * q))
        C_RB_nu[3] = (Izz - Iyy) * q * r + m * (yg * (-u * q + v * p) - zg * (-w * p + u * r))
        C_RB_nu[4] = (Ixx - Izz) * p * r + m * (zg * (-v * r + w * q) - xg * (-u * q + v * p))
        C_RB_nu[5] = (Iyy - Ixx) * p * q + m * (xg * (-w * p + u * r) - yg * (-v * r + w * q))
        
        # Added Mass Coriolis C_A(nu)*nu
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
        """Calculates tau_control based on inputs."""
        u = nu[0, 0] 
        
        throttle_cmd = control_cmds['throttle'] # -1.0 (fwd), 0.5 (rev)
        yaw_cmd = control_cmds['yaw'] # 1.0 (left), -1.0 (right)
        pitch_cmd = control_cmds['pitch'] # 1.0 (down), -1.0 (up)

        # Thrust
        X_thrust = throttle_cmd * self.MAX_THRUST
        
        # Rudder (Yaw)
        delta_r = yaw_cmd * self.MAX_RUDDER_ANGLE
        Y_rudder = self.params['Yuu_delta_r'] * (abs(u) * u) * delta_r
        N_rudder = self.params['Nuu_delta_r'] * (abs(u) * u) * delta_r
        
        # Stern Plane (Pitch)
        delta_s = pitch_cmd * self.MAX_STERN_ANGLE
        Z_stern = self.params['Zuu_delta_s'] * (abs(u) * u) * delta_s
        M_stern = self.params['Muu_delta_s'] * (abs(u) * u) * delta_s
        
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
        tau_G = self.calculate_hydrostatics(self.eta)
        tau_D = self.calculate_damping(self.nu)
        tau_C = self.calculate_coriolis(self.nu)
        tau_ctrl = self.calculate_control_forces(self.nu, control_cmds)
        
        tau_total = tau_ctrl - tau_G - tau_D - tau_C
        
        nu_dot = self.M_inv @ tau_total
        
        self.nu = self.nu + nu_dot * dt
        
        theta = self.eta[4, 0]
        if abs(np.cos(theta)) < 1e-6:
             self.eta[4, 0] = np.sign(theta) * (np.pi/2 - 1e-3) # nudge
        
        J = jacobian(self.eta)
        eta_dot = J @ self.nu
        
        self.eta = self.eta + eta_dot * dt
        
        if self.eta[2, 0] > 0:
            self.eta[2, 0] = 0
            if self.nu[2, 0] < 0:
                self.nu[2, 0] = 0
            if eta_dot[2, 0] > 0:
                self.nu = inv(J) @ np.array([eta_dot[0,0], eta_dot[1,0], 0, eta_dot[3,0], eta_dot[4,0], eta_dot[5,0]]).reshape(6,1)