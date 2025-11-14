import numpy as np
from numpy.linalg import inv
from auv_parameters import REMUS_PARAMS, PARAMS_DERIVED
from auv_hydrodynamic_parameters import HYDRO_PARAMS

# Helper Functions

def skew(v):
    """ Converts a 3-element vector into a 3x3 skew-symmetric matrix"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def jacobian(eta):
    """
    Computes the 6-DOF transformation matrix J(eta) from body-frame to world-frame.
    
    This matrix is essential for converting the AUV's velocities from its
    own moving reference frame (body-frame, 'nu') to the fixed
    global reference frame (world-frame, 'eta_dot').
    
    The 'eta' vector contains the AUV's world position (x, y, z) and
    its orientation as Euler angles (phi, theta, psi).
    """
    
    # Unpack the Euler angles from the 'eta' state vector
    phi, theta, psi = eta[3:6].flatten()
    
    # Pre-calculate sine and cosine of the angles for efficiency
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    
    # J1 is the standard ZYX Euler 3x3 rotation matrix for linear velocities
    # It transforms [u, v, w] (body-frame) to [x_dot, y_dot, z_dot] (world-frame)
    J1 = np.array([
        [cpsi*cth, -spsi*cphi + cpsi*sth*sphi, spsi*sphi + cpsi*sth*cphi],
        [spsi*cth, cpsi*cphi + spsi*sth*sphi, -cpsi*sphi + spsi*sth*cphi],
        [-sth, cth*sphi, cth*cphi]
    ])
    
    # J2 is the 3x3 transformation matrix for angular velocities
    # It transforms [p, q, r] (body-frame) to [phi_dot, theta_dot, psi_dot] (world-frame)
    # This matrix is not a simple rotation and has a singularity
    # when pitch (theta) is +/- 90 degrees
    J2 = np.array([
        [1, sphi*np.tan(theta), cphi*np.tan(theta)],
        [0, cphi, -sphi],
        [0, sphi/cth, cphi/cth]
    ])
    
    # Combine J1 and J2 into the full 6x6 Jacobian matrix
    J = np.zeros((6, 6))
    J[0:3, 0:3] = J1
    J[3:6, 3:6] = J2
    return J

# Physics Model Class

class AUVPhysicsModel:
    """
    Implements the full 6-DOF (Degrees of Freedom) dynamic model for the AUV.
    
    This class represents the "engine" of the simulator. It solves the
    complete Newton-Euler equations of motion, which are expressed in
    the standard matrix form:
    M * nu_dot + C(nu) * nu + D(nu) * nu + g(eta) = tau
    
    The origin of the body-fixed coordinate system is set at the
    vehicle's Center of Buoyancy (CB).
    """
    def __init__(self):
        # Load all physical and hydrodynamic parameters from the other files
        self.params = {**REMUS_PARAMS, **HYDRO_PARAMS}
        
        # State Vectors
        # 'eta' is the 6-DOF position and orientation in the world-frame (Earth-fixed)
        # [x, y, z, phi(roll), theta(pitch), psi(yaw)]
        self.eta = np.zeros((6, 1))
        self.eta[2] = -5.0 # Start the AUV 5 meters deep (negative Z is down)
        
        # 'nu' is the 6-DOF linear and angular velocities in the body-frame
        # [u(surge), v(sway), w(heave), p(roll), q(pitch), r(yaw)]
        self.nu = np.zeros((6, 1))
        
        self.m = self.params['m'] # Mass (kg)
        self.L = self.params['L'] # Length (m)
        self.W = self.params['W'] # Weight (N)
        self.B = self.params['B'] # Buoyancy (N)
        self.g = 9.81 # Acceleration due to gravity (m/s^2)

        # Coordinate System Definition
        # The origin (0,0,0) of our body-frame is the Center of Buoyancy (CB)
        # Therefore, the vector from the origin to the CB is zero
        self.r_b = np.array([0., 0., 0.])
        
        # We must find the location of the Center of Gravity (CG) relative to our
        # new origin (CB). We do this by finding both points relative to the
        # nose of the AUV (from auv_parameters.py) and then finding the
        # vector difference between them.
        r_b_vec_nose = np.array(PARAMS_DERIVED["cb_pos"])
        r_g_vec_nose = np.array(PARAMS_DERIVED["cg_pos"])
        
        # self.r_g is the crucial vector from the origin (CB) to the CG
        # This offset is the source of the AUV's hydrostatic stability
        self.r_g = (r_g_vec_nose - r_b_vec_nose).reshape(3, 1)
        
        # Define the 3x3 inertia tensor (moments of inertia) about the origin
        # We assume the AUV is symmetrical, so the cross-products are zero
        self.I_o = np.diag([
            self.params['Ixx'],
            self.params['Iyy'],
            self.params['Izz']
        ])
        
        # Pre-calculate the system matrices
        self.build_mass_matrices()
        self.build_damping_matrices()
        
        # Define the physical limits of the AUV's actuators
        self.MAX_THRUST = 40.0 # Newtons
        self.MAX_RUDDER_ANGLE = np.radians(20) # 20 degrees in radians
        self.MAX_STERN_ANGLE = np.radians(20) # 20 degrees in radians

    def build_mass_matrices(self):
        """
        Builds the 6x6 system inertia matrix 'M' and its inverse 'M_inv'.
        
        The total mass matrix M is the sum of the rigid-body mass (M_RB)
        and the hydrodynamic "added mass" (M_A).
        """
        m = self.m
        
        # Rigid-Body Mass Matrix (M_RB)
        # This matrix represents the AUV's physical mass and inertia.
        # Because our origin (CB) is not at the Center of Gravity (CG),
        # the matrix is NOT diagonal. The offset 'r_g' creates
        # off-diagonal terms that couple linear and angular motion.
        S_rg = skew(self.r_g.flatten())
        
        self.M_RB = np.zeros((6, 6))
        self.M_RB[0:3, 0:3] = np.diag([m, m, m]) # Top-left: mass
        self.M_RB[0:3, 3:6] = -m * S_rg # Top-right: coupling
        self.M_RB[3:6, 0:3] = m * S_rg # Bottom-left: coupling
        self.M_RB[3:6, 3:6] = self.I_o # Bottom-right: inertia tensor
        
        # Added Mass Matrix (M_A)
        # This matrix represents the inertia of the water that the AUV must
        # push out of the way when it accelerates. It makes the AUV "feel"
        # heavier than it actually is.
        self.M_A = np.diag([
            -self.params['Xu_dot'], # Surge added mass
            -self.params['Yv_dot'], # Sway added mass
            -self.params['Zw_dot'], # Heave added mass
            -self.params['Kp_dot'], # Roll added mass
            -self.params['Mq_dot'], # Pitch added mass
            -self.params['Nr_dot'] # Yaw added mass
        ])
        # Add any defined off-diagonal added mass terms (couplings)
        self.M_A[1, 5] = self.M_A[5, 1] = -self.params.get('Y_r_dot', 0)
        
        # Total Mass Matrix (M)
        # The final system inertia is the sum of the AUV's physical mass
        # and the hydrodynamic added mass.
        self.M = self.M_RB + self.M_A
        
        # We pre-calculate the inverse of M. This is a huge optimization,
        # as inverting a matrix is computationally expensive, and we would
        # otherwise have to do it in every single step of the simulation.
        self.M_inv = inv(self.M)

    def build_damping_matrices(self):
        """
        Builds the matrices that represent hydrodynamic damping (drag).
        
        Damping is split into two parts: linear (proportional to velocity)
        and quadratic (proportional to velocity-squared).
        """
        # Linear Damping (D_lin)
        # This models viscous friction, which dominates at low speeds.
        # We create a 6x6 diagonal matrix of the linear damping coefficients.
        self.D_lin = -np.diag([
            self.params['Xu'],
            self.params['Yv'],
            self.params['Zw'],
            self.params['Kp'],
            self.params['Mq'],
            self.params['Nr']
        ])
        
        # Quadratic Damping (D_quad_coeffs)
        # This models turbulent drag, which dominates at higher speeds.
        # We only store the coefficients. The full force (v * |v|)
        # will be calculated in the 'calculate_damping' function.
        self.D_quad_coeffs = -np.diag([
            self.params['X_u|u|'],
            self.params['Y_v|v|'],
            self.params['Z_w|w|'],
            self.params['K_p|p|'],
            self.params['M_q|q|'],
            self.params['N_r|r|']
        ])

    def reset(self):
        """ Resets the AUV state back to its starting position"""
        self.eta = np.zeros((6, 1))
        self.eta[2] = -5.0 # Reset to 5m deep
        self.nu = np.zeros((6, 1)) # Reset all velocities to zero

    def calculate_hydrostatics(self, eta):
        """
        Calculates the g(eta) vector: the restoring forces and torques.
        
        These are the passive forces from Weight (Gravity) and Buoyancy
        that naturally try to bring the AUV to a stable, upright
        position at the surface.
        """
        # Get the AUV's current orientation
        phi, theta, psi = eta[3:6].flatten()
        
        # Get the rotation matrix to transform from world to body frame.
        # We need this to determine the direction of "down" relative
        # to the AUV's tilted body.
        R_b_w = jacobian(eta)[0:3, 0:3]
        
        # Gravity (Weight) Force
        # Weight (self.W) always acts straight down [0, 0, W] in the world frame.
        f_g_world = np.array([[0], [0], [self.W]])
        # Convert this "down" force into the AUV's body-frame components.
        f_g_body = R_b_w.T @ f_g_world
        # Calculate the torque (moment) this force creates.
        # Torque = r x F. This is calculated using the lever arm self.r_g
        # (the vector from the origin to the CG).
        tau_g_body = skew(self.r_g.flatten()) @ f_g_body
    
        # Buoyancy Force
        # Buoyancy (self.B) always acts straight up [0, 0, -B] in the world frame.
        f_b_world = np.array([[0], [0], [-self.B]])
        # Convert this "up" force into the AUV's body-frame components.
        f_b_body = R_b_w.T @ f_b_world
        # Calculate the torque. Since Buoyancy acts at the origin (CB),
        # its lever arm is [0,0,0], so the torque is always zero.
        tau_b_body = np.zeros((3, 1))
        
        # The final g(eta) vector is the sum of all hydrostatic forces and torques.
        # Since B > W, the net force will be upwards, making the AUV float.
        # Since r_g is offset, a roll/pitch will create a restoring torque.
        g_eta = np.vstack((f_g_body + f_b_body, tau_g_body + tau_b_body))
        return g_eta

    def calculate_damping(self, nu):
        """
        Calculates the D(nu)*nu vector: the total damping (drag) forces.
        
        This function combines both linear and quadratic damping effects.
        """
        # First, calculate the linear damping force
        D_nu = self.D_lin @ nu
        
        # Next, calculate the quadratic damping force.
        # We use (np.abs(nu) * nu) to get a sign-safe v*|v| operation.
        # This ensures the drag force always opposes the velocity.
        D_nu_quad = self.D_quad_coeffs @ (np.abs(nu) * nu)
        
        # The total damping is the sum of both components.
        return D_nu + D_nu_quad

    def calculate_coriolis(self, nu):
        """
        Calculates the C(nu)*nu vector: Coriolis and centripetal forces.
        
        These are "fictitious" forces that appear because we are
        in a rotating reference frame. They are split into two parts:
        C_RB (from the rigid body) and C_A (from the added mass).
        """
        # Rigid-Body Coriolis C_RB(nu)*nu
        # These are the terms from the Newton-Euler equations that
        # describe motion coupling (e.g., 'm * (-v * r + w * q)').
        
        # Unpack all state variables for use in the equations
        u, v, w, p, q, r = nu.flatten()
        m = self.m
        xg, yg, zg = self.r_g.flatten()
        Ixx, Iyy, Izz = self.params['Ixx'], self.params['Iyy'], self.params['Izz']
        
        # Pre-allocate the 6x1 C_RB vector
        C_RB_nu = np.zeros((6, 1))
        
        # Each line below directly implements the 6-DOF Newton-Euler equations
        # for Coriolis and centripetal forces.

        # SURGE (x-force)
        C_RB_nu[0] = m * (-v * r + w * q - xg * (q**2 + r**2) + yg * (p * q) + zg * (p * r))

        # SWAY (y-force)
        C_RB_nu[1] = m * (-w * p + u * r - yg * (p**2 + r**2) + zg * (q * r) + xg * (p * q))

        # HEAVE (z-force)
        C_RB_nu[2] = m * (-u * q + v * p - zg * (p**2 + q**2) + xg * (r * p) + yg * (r * q))

        # ROLL (K-torque)
        C_RB_nu[3] = (Izz - Iyy) * q * r + m * (yg * (-u * q + v * p) - zg * (-w * p + u * r))

        # PITCH (M-torque)
        C_RB_nu[4] = (Ixx - Izz) * p * r + m * (zg * (-v * r + w * q) - xg * (-u * q + v * p))

        # YAW (N-torque)
        C_RB_nu[5] = (Iyy - Ixx) * p * q + m * (xg * (-w * p + u * r) - yg * (-v * r + w * q))
        
        # Added Mass Coriolis C_A(nu)*nu
        # This is a simplified model of the Coriolis forces caused by
        # the "added mass" of water being moved by the AUV.
        A11, A22, A33 = self.M_A[0,0], self.M_A[1,1], self.M_A[2,2]
        A44, A55, A66 = self.M_A[3,3], self.M_A[4,4], self.M_A[5,5]

        C_A_nu = np.zeros((6, 1))
        C_A_nu[0] = (A33*w*q - A22*v*r)
        C_A_nu[1] = (A11*u*r - A33*w*p)
        C_A_nu[2] = (A22*v*p - A11*u*q)
        C_A_nu[3] = (A66*r*q - A55*q*r)
        C_A_nu[4] = (A44*p*r - A66*r*p)
        C_A_nu[5] = (A55*q*p - A44*p*q)
        
        # The total Coriolis vector is the sum of both parts.
        return C_RB_nu + C_A_nu


    def calculate_control_forces(self, nu, control_cmds):
        """
        Calculates the 'tau' vector: the forces from thrusters and fins.
        
        This function translates high-level commands (like 'throttle'
        or 'yaw') into physical forces and torques.
        """
        # Get the AUV's current forward speed (surge velocity)
        u = nu[0, 0] 
        
        # Unpack the commands from the dictionary
        throttle_cmd = control_cmds['throttle'] # -1.0 (fwd) to +0.5 (rev)
        yaw_cmd = control_cmds['yaw'] # 1.0 (left) to -1.0 (right)
        pitch_cmd = control_cmds['pitch'] # 1.0 (down) to -1.0 (up)

        # Thrust Force (Surge, X-axis)
        # This is the force from the main propeller.
        X_thrust = throttle_cmd * self.MAX_THRUST
        
        # Rudder Forces (Yaw, N-axis and Sway, Y-axis)
        # Calculate the rudder angle from the command
        delta_r = yaw_cmd * self.MAX_RUDDER_ANGLE
        # The effectiveness of fins is proportional to the square of water speed.
        # We use (abs(u) * u) to ensure the force direction is correct
        # even when the AUV is moving in reverse.
        Y_rudder = self.params['Yuu_delta_r'] * (abs(u) * u) * delta_r
        N_rudder = self.params['Nuu_delta_r'] * (abs(u) * u) * delta_r
        
        # Stern Plane Forces (Pitch, M-axis and Heave, Z-axis)
        # Calculate the stern plane angle from the command
        delta_s = pitch_cmd * self.MAX_STERN_ANGLE
        # Like the rudder, effectiveness is proportional to u*|u|.
        Z_stern = self.params['Zuu_delta_s'] * (abs(u) * u) * delta_s
        M_stern = self.params['Muu_delta_s'] * (abs(u) * u) * delta_s
        
        # Roll Force (K-axis)
        # This AUV model is not actuated in roll (no roll fins).
        K_control = 0.0
        
        # Assemble all 6 forces and torques into the final 'tau' vector.
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
        """ Advances the simulation by one time step 'dt'"""
        
        # First, calculate all four major force/torque vectors for
        # the AUV's current state (eta, nu).
        tau_G = self.calculate_hydrostatics(self.eta)
        tau_D = self.calculate_damping(self.nu)
        tau_C = self.calculate_coriolis(self.nu)
        tau_ctrl = self.calculate_control_forces(self.nu, control_cmds)
        
        # Sum all forces to get the total net force on the AUV.
        # We subtract G, D, and C because they are on the left side
        # of the standard equation (M*nu_dot + C + D + g = tau),
        # so we move them to the right: M*nu_dot = tau - C - D - g
        tau_total = tau_ctrl - tau_G - tau_D - tau_C
        
        # Solve for Acceleration
        # This is the core of the simulation. We solve for the
        # body-frame acceleration (nu_dot) by multiplying the total
        # force by the inverse of the mass matrix.
        # This is the 6-DOF equivalent of 'a = F / m'.
        nu_dot = self.M_inv @ tau_total
        
        # Integrate to Find New Velocity
        # Use simple Euler integration: new_velocity = old_velocity + accel * dt
        self.nu = self.nu + nu_dot * dt
        
        # Integrate to Find New Position
        
        # Safety check for gimbal lock. The 'jacobian' matrix (J2)
        # has a tan(theta) term, which becomes infinite if pitch (theta)
        # is exactly +/- 90 degrees. This code nudges the angle slightly
        # away from 90 to prevent a mathematical crash.
        theta = self.eta[4, 0]
        if abs(np.cos(theta)) < 1e-6:
             self.eta[4, 0] = np.sign(theta) * (np.pi/2 - 1e-3)
        
        # Use the jacobian to transform body-frame velocities 'nu'
        # into world-frame velocities 'eta_dot'.
        J = jacobian(self.eta)
        eta_dot = J @ self.nu
        
        # Use Euler integration to find the new world-frame position 'eta'.
        # new_position = old_position + world_velocity * dt
        self.eta = self.eta + eta_dot * dt
        
        # Surface Interaction Physics
        # Prevent the AUV from going above the water surface (Z=0).
        if self.eta[2, 0] > 0:
            self.eta[2, 0] = 0 # Clamp position to the surface
            
            # Also kill any upward velocity to prevent "bouncing"
            if self.nu[2, 0] < 0: # w is positive down, so < 0 is upward
                self.nu[2, 0] = 0
            
            # More complex: stop world-frame z-velocity
            if eta_dot[2, 0] > 0:
                # We must recalculate 'nu' to reflect this new world velocity
                eta_dot[2, 0] = 0
                self.nu = inv(J) @ eta_dot