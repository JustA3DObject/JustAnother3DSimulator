import numpy as np
from auv_physics import AUVPhysicsModel
from auv_visuals import AUVGeometry

class AUVController: 
    """ Connects user input to the physics model and visual geometry"""
    def __init__(self):
        # Create the physics model and visual geometry
        self.physics = AUVPhysicsModel()
        self.geometry = AUVGeometry()

        # Simulation timestep
        self.dt = 0.05 # s

        # Key tracking
        self.keys_pressed = set()

    def update_state(self):
        """
        Update the state by sending commands to the physics model
        and then reading its new state
        """
        
        # Parse keyboard inputs into control commands
        control_cmds = {
            'throttle': 0.0,
            'yaw': 0.0,
            'pitch': 0.0
        }
        
        if 'r' in self.keys_pressed:
            self.physics.reset()

        # 'W' = Forward = Negative X-direction = Negative Thrust
        if 'w' in self.keys_pressed:
            control_cmds['throttle'] = -1.0  # Full forward thrust
        # 'X' = Backward = Positive X-direction = Positive Thrust
        elif 'x' in self.keys_pressed:
            control_cmds['throttle'] = 0.5 # Half reverse thrust
            
        # Brake (applies counter-thrust)
        if 'b' in self.keys_pressed:
            if self.physics.nu[0,0] < -0.1: # If moving forward (u < 0)
                control_cmds['throttle'] = 1.0 # Apply reverse (positive)
            elif self.physics.nu[0,0] > 0.1: # If moving backward (u > 0)
                control_cmds['throttle'] = -1.0 # Apply forward (negative)

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

    def get_plot_assets(self):
        """
        Gets the current position/orientation from the physics model
        and returns all transformed geometry assets for plotting
        """
        pos = self.physics.eta[0:3].flatten()
        ori = self.physics.eta[3:6].flatten()
        return self.geometry.get_transformed_assets(pos, ori)

    def get_state_text(self):
        """
        Returns a formatted string of the current physics state for display
        """
        pos = self.physics.eta[0:3].flatten()
        vel = self.physics.nu.flatten()
        ori = self.physics.eta[3:6].flatten()
        
        speed = abs(vel[0]) 
        
        return (
            f"Pos (x,y,z): {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}\n"
            f"Vel (u,v,w): {vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f} m/s\n"
            f"Ori (r,p,y): {np.degrees(ori[0]):.1f}, {np.degrees(ori[1]):.1f}, {np.degrees(ori[2]):.1f}°\n"
            f"AngVel (p,q,r): {np.degrees(vel[3]):.1f}, {np.degrees(vel[4]):.1f}, {np.degrees(vel[5]):.1f}°/s"
        )