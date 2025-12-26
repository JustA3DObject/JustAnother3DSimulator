import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# Import the main controller
from auv_controller import AUVController

def create_interactive_auv():
    """ Create interactive AUV with keyboard controls"""
    
    # Create controller (which creates the physics and visuals)
    controller = AUVController()
    
    # Setup figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Store surface objects
    plot_artists = {
        'nose': None, 'mid': None, 'tail': None, 'sss1': None, 'sss2': None,
        'mast': None, 'cage': None, 'dvl_collection': None,
        'fin_collections': [], 'prop_surfaces': []
    }
    
    # Control instructions
    instructions = (
        " 6-DOF DYNAMIC SIMULATOR\n"
        "W - Throttle Forward\n"
        "X - Throttle Backward\n"
        "A - Yaw Left | D - Yaw Right\n"
        "Z - Pitch Up | C - Pitch Down\n"
        "B - Brake\n"
        "R - Reset\n"
        "AUV is positively buoyant and hydrostatically stable."
    )
    
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
        pos = controller.physics.eta[0:3].flatten()
        ax.set_xlim(pos[0] - limit, pos[0] + limit)
        ax.set_ylim(pos[1] - limit, pos[1] + limit)
        ax.set_zlim(pos[2] - limit, pos[2] + limit)
        
        return []
    
    def update(frame):
        """Animation update function"""
        
        # Run the controller and physics
        controller.update_state()
        
        # Clear previous artists
        for artist in plot_artists.values():
            if isinstance(artist, list):
                for item in artist:
                    item.remove()
                artist.clear()
            elif artist:
                artist.remove()

        # Get new plot assets from controller
        assets = controller.get_plot_assets()
        
        # Plot all assets
        plot_artists['nose'] = ax.plot_surface(*assets['nose'], color='blue', alpha=0.7, rstride=5, cstride=5)
        plot_artists['mid'] = ax.plot_surface(*assets['mid'], color='green', alpha=0.7, rstride=5, cstride=5)
        plot_artists['tail'] = ax.plot_surface(*assets['tail'], color='red', alpha=0.7, rstride=5, cstride=5)
        plot_artists['sss1'] = ax.plot_surface(*assets['sss1'], color='grey', alpha=0.8)
        plot_artists['sss2'] = ax.plot_surface(*assets['sss2'], color='grey', alpha=0.8)
        plot_artists['mast'] = ax.plot_surface(*assets['mast'], color='silver', alpha=0.9)
        plot_artists['cage'] = ax.plot_surface(*assets['cage'], color='grey', alpha=0.4)
        
        plot_artists['dvl_collection'] = Poly3DCollection(assets['dvl_faces'], facecolors='orange', alpha=0.8)
        ax.add_collection3d(plot_artists['dvl_collection'])
        
        # assets['fins'] is a list of fins, where each fin is a list of faces (polygons).
        # Poly3DCollection takes a list of polygons to form a single 3D object.
        for fin_faces in assets['fins']:
            fin_col = Poly3DCollection(fin_faces, facecolors='darkslategrey', edgecolors='black', alpha=0.9)
            ax.add_collection3d(fin_col)
            plot_artists['fin_collections'].append(fin_col)
        
        for X_prop, Y_prop, Z_prop in assets['prop_blades']:
            prop_surf = ax.plot_surface(X_prop, Y_prop, Z_prop, color='black', alpha=0.9)
            plot_artists['prop_surfaces'].append(prop_surf)
        
        # Update camera and text
        pos = controller.physics.eta[0:3].flatten()
        offset = 4
        ax.set_xlim(pos[0] - offset, pos[0] + offset)
        ax.set_ylim(pos[1] - offset, pos[1] + offset)
        ax.set_zlim(pos[2] - offset, pos[2] + offset)
        
        state_text.set_text(controller.get_state_text())
        
        # Return all artists for blitting
        all_artists = [plot_artists[k] for k in plot_artists if k not in ['fin_collections', 'prop_surfaces']]
        all_artists += plot_artists['fin_collections'] + plot_artists['prop_surfaces']
        return all_artists
    
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