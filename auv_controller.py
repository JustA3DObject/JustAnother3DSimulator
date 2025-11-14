import numpy as np
from auv_physics import AUVPhysicsModel
from auv_visuals import AUVGeometry

class AUVController: 
    """ Connects user input to the physics model and visual geometry"""
    def __init__(self):
        self.physics = AUVPhysicsModel()
        self.geometry = AUVGeometry()

        # Define the simulation's discrete time step in seconds.
        # This value dictates how far in the future the physics engine
        # calculates on each animation frame. A smaller value is more
        # accurate but more computationally intensive.
        self.dt = 0.05 # 50 milliseconds

        # Create an empty set to keep track of which keys are currently
        # being held down. A set is very efficient for this, as it
        # provides instant add/discard/lookup and automatically
        # handles duplicate key-down events.
        self.keys_pressed = set()

    def update_state(self):
        """
        This is the main function of the controller, called once per
        animation frame by the main simulator loop.
        
        It parses all current user input and tells the physics model
        to advance one step in time.
        """
        
        # At the beginning of each frame, we create a "clean slate"
        # dictionary of control commands. All commands default to zero.
        control_cmds = {
            'throttle': 0.0,  # No surge force
            'yaw': 0.0,       # No rudder deflection
            'pitch': 0.0      # No stern plane deflection
        }
        
        # Check if the 'r' (reset) key is currently being pressed.
        if 'r' in self.keys_pressed:
            self.physics.reset()

        # Throttle Logic
        # The AUV's geometry is defined with the x-axis pointing from
        # nose to tail, so "forward" is motion in the negative X direction.
        
        # Check if 'w' (forward) is pressed.
        if 'w' in self.keys_pressed:
            # Set the throttle command to -1.0 (full power forward).
            # The physics model will turn this into a negative X-force.
            control_cmds['throttle'] = -1.0
        # Otherwise, check if 'x' (backward) is pressed.
        elif 'x' in self.keys_pressed:
            # Set the throttle command to 0.5 (half power reverse).
            # The physics model will turn this into a positive X-force.
            control_cmds['throttle'] = 0.5
            
        # Brake Logic
        # The brake applies a counter-thrust to slow the AUV down.
        if 'b' in self.keys_pressed:
            # Check the AUV's actual current surge velocity (u), which is nu[0].
            if self.physics.nu[0,0] < -0.1: # If moving forward (u is negative)
                # Apply a full reverse (positive) thrust to stop.
                control_cmds['throttle'] = 1.0
            elif self.physics.nu[0,0] > 0.1: # If moving backward (u is positive)
                # Apply a full forward (negative) thrust to stop.
                control_cmds['throttle'] = -1.0

        # Yaw (Rudder) Logic
        if 'a' in self.keys_pressed: # 'a' for yaw left
            # Set the yaw command to +1.0.
            control_cmds['yaw'] = 1.0
        if 'd' in self.keys_pressed: # 'd' for yaw right
            # Set the yaw command to -1.0 (negative rudder deflection).
            control_cmds['yaw'] = -1.0
        
        # Pitch (Stern Plane) Logic
        if 'z' in self.keys_pressed: # 'z' for pitch (nose) up
            control_cmds['pitch'] = -1.0
        if 'c' in self.keys_pressed: # 'c' for pitch (nose) down
            control_cmds['pitch'] = 1.0
    
        self.physics.step(self.dt, control_cmds)

    def get_plot_assets(self):
        """
        This is a helper function for the main visualization loop.
        
        It retrieves the AUV's current state from the physics model
        and asks the geometry object to prepare all 3D assets for plotting.
        """
        # Get the 6-DOF world-frame position vector 'eta'
        # and extract the [x, y, z] components.
        pos = self.physics.eta[0:3].flatten()
        # Extract the [roll, pitch, yaw] orientation components.
        ori = self.physics.eta[3:6].flatten()
        
        # Pass the current position and orientation to the geometry object.
        # This will take the base AUV mesh, rotate it, and move it to
        # its correct location in the 3D world, returning all the
        # transformed vertices ready for matplotlib to draw.
        return self.geometry.get_transformed_assets(pos, ori)

    def get_state_text(self):
        """ Returns a formatted string of the current physics state"""
        # Get the full state vectors from the physics model
        pos = self.physics.eta[0:3].flatten()
        vel = self.physics.nu.flatten()
        ori = self.physics.eta[3:6].flatten()
        
        speed = abs(vel[0]) 
        
        # Format all the state information into a multi-line string.
        # We convert radians to degrees for easier readability.
        return (
            f"Pos (x,y,z): {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}\n"
            f"Vel (u,v,w): {vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f} m/s\n"
            f"Ori (r,p,y): {np.degrees(ori[0]):.1f}, {np.degrees(ori[1]):.1f}, {np.degrees(ori[2]):.1f}°\n"
            f"AngVel (p,q,r): {np.degrees(vel[3]):.1f}, {np.degrees(vel[4]):.1f}, {np.degrees(vel[5]):.1f}°/s"
        )