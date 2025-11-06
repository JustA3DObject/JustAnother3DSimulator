import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Import for plotting 3D polygons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_auv(a, a_offset, b, c, c_offset, n, theta_tail, d, lf, l):
    """
    Generates and plots a 3D model and 2D profile of a torpedo-shaped AUV,
    including simplified control fins, propeller hub, and external sensors.

    ... (Args list remains the same) ...
    """

    # --- 1. Define Basic Parameters and Resolution ---
    
    r_max = d / 2
    num_x_points = 100
    num_theta_points = 80
    theta = np.linspace(0, 2 * np.pi, num_theta_points)
    # Prevent Z-fighting (stitching) by adding a tiny offset
    z_offset = 0.001 

    # --- 2. Calculate Coordinates for each Section ---

    # == NOSE SECTION ==
    x_nose = np.linspace(0, a, num_x_points)
    # Equation: r(x) = r_max - (r_max - a_offset) * (1 - x/a)^n
    # This gives r=a_offset at x=0 and r=r_max at x=a.
    r_nose = r_max - (r_max - a_offset) * (1 - x_nose / a)**n
    
    X_nose, THETA_nose = np.meshgrid(x_nose, theta)
    R_nose, _ = np.meshgrid(r_nose, theta)
    Y_nose = R_nose * np.cos(THETA_nose)
    Z_nose = R_nose * np.sin(THETA_nose)

    # == MID-SECTION (Cylinder) ==
    mid_section_length = lf - a
    num_x_mid_points = max(2, int(num_x_points * (mid_section_length / a)))
    x_mid = np.linspace(a, lf, num_x_mid_points)
    r_mid = np.full_like(x_mid, r_max)
    
    X_mid, THETA_mid = np.meshgrid(x_mid, theta)
    R_mid, _ = np.meshgrid(r_mid, theta)
    Y_mid = R_mid * np.cos(THETA_mid)
    Z_mid = R_mid * np.sin(THETA_mid)

    # == TAIL SECTION (Power Series Curve of Revolution) ==
    num_x_tail_points = max(2, int(num_x_points * (c / a)))
    x_tail = np.linspace(lf, l, num_x_tail_points)
    
    # MODIFIED Tail Equation to use c_offset
    # Tapers from r_max (at x=lf) down to c_offset (at x=l)
    x_norm_tail = (x_tail - lf) / c
    # Use 1-x^n to get the curve shape, scaling between (r_max-c_offset)
    r_tail = c_offset + (r_max - c_offset) * (1 - x_norm_tail**n)
    
    X_tail, THETA_tail = np.meshgrid(x_tail, theta)
    R_tail, _ = np.meshgrid(r_tail, theta)
    Y_tail = R_tail * np.cos(THETA_tail)
    Z_tail = R_tail * np.sin(THETA_tail)
    
    r_final = c_offset # Final radius at the tail

    # --- 3. Define External Components ---

    # == Side-Scan Sonar (SSS) Patches ==
    # On the mid-body, port (+Y) and starboard (-Y)
    sss_len = 0.3
    sss_width_angle = 0.1 # Angular width
    x_sss_start = a + (mid_section_length - sss_len) / 2
    x_sss_end = x_sss_start + sss_len
    
    x_sss = np.linspace(x_sss_start, x_sss_end, 10)
    
    # SSS 1 (Starboard, -Y side, around theta=pi)
    th_sss1 = np.linspace(np.pi - sss_width_angle, np.pi + sss_width_angle, 10)
    X_sss1, TH_sss1 = np.meshgrid(x_sss, th_sss1)
    R_sss1 = r_max + z_offset # Place just outside the hull
    Y_sss1 = R_sss1 * np.cos(TH_sss1)
    Z_sss1 = R_sss1 * np.sin(TH_sss1)

    # SSS 2 (Port, +Y side, around theta=0)
    th_sss2 = np.linspace(-sss_width_angle, sss_width_angle, 10)
    X_sss2, TH_sss2 = np.meshgrid(x_sss, th_sss2)
    R_sss2 = r_max + z_offset
    Y_sss2 = R_sss2 * np.cos(TH_sss2)
    Z_sss2 = R_sss2 * np.sin(TH_sss2)

    # == DVL (Doppler Velocity Log) Box ==
    # On the underside (-Z) near the tail junction
    dvl_len = 0.1
    dvl_width = 0.08
    dvl_height = 0.03
    
    x_dvl_start = lf - dvl_len - 0.05 # 5cm fwd of tail junction
    x_dvl_end = x_dvl_start + dvl_len
    y_dvl_half = dvl_width / 2
    z_dvl_top = -r_max # Attaches to hull bottom
    z_dvl_bottom = z_dvl_top - dvl_height
    
    # Define 8 vertices of the DVL cuboid
    v = np.array([
        [x_dvl_start, -y_dvl_half, z_dvl_top],    # v0
        [x_dvl_end, -y_dvl_half, z_dvl_top],      # v1
        [x_dvl_end, y_dvl_half, z_dvl_top],       # v2
        [x_dvl_start, y_dvl_half, z_dvl_top],     # v3
        [x_dvl_start, -y_dvl_half, z_dvl_bottom], # v4
        [x_dvl_end, -y_dvl_half, z_dvl_bottom],   # v5
        [x_dvl_end, y_dvl_half, z_dvl_bottom],    # v6
        [x_dvl_start, y_dvl_half, z_dvl_bottom]   # v7
    ])
    
    # Define the 6 faces (lists of vertex indices)
    dvl_faces = [
        [v[0], v[1], v[2], v[3]], # Top
        [v[4], v[5], v[6], v[7]], # Bottom
        [v[0], v[1], v[5], v[4]], # Front
        [v[2], v[3], v[7], v[6]], # Back
        [v[0], v[3], v[7], v[4]], # Left
        [v[1], v[2], v[6], v[5]]  # Right
    ]
    dvl_collection = Poly3DCollection(dvl_faces, facecolors='orange')

    # == Antenna Mast ==
    # On the top (+Z) of the rear mid-section
    mast_height = 0.08
    mast_radius = 0.01
    x_mast_base = lf - 0.1 # 10cm fwd of tail junction
    
    theta_mast = np.linspace(0, 2 * np.pi, 20)
    z_mast = np.linspace(r_max, r_max + mast_height, 2)
    
    TH_mast, Z_mast = np.meshgrid(theta_mast, z_mast)
    X_mast = x_mast_base + mast_radius * np.cos(TH_mast)
    Y_mast = 0 + mast_radius * np.sin(TH_mast)
    
    # --- 3.5. Define Control Fins and Propeller ---
    
    # == CONTROL FINS ==
    fin_length = 0.12
    fin_span = 0.1
    fin_taper_ratio = 0.8
    fin_x_start = l - fin_length
    fin_x_end = l
    
    x_norm_fin_start = (fin_x_start - lf) / c
    r_fin_start = c_offset + (r_max - c_offset) * (1 - x_norm_fin_start**n)
    r_fin_end = r_final
    
    fin_verts = []
    # Fin 1: Horizontal (+Y)
    v1 = [fin_x_start, r_fin_start, 0]; v2 = [fin_x_start, r_fin_start + fin_span, 0]
    v3 = [fin_x_end, r_fin_end + fin_span * fin_taper_ratio, 0]; v4 = [fin_x_end, r_fin_end, 0]
    fin_verts.append([v1, v2, v3, v4])
    # Fin 2: Horizontal (-Y)
    v1 = [fin_x_start, -r_fin_start, 0]; v2 = [fin_x_start, -(r_fin_start + fin_span), 0]
    v3 = [fin_x_end, -(r_fin_end + fin_span * fin_taper_ratio), 0]; v4 = [fin_x_end, -r_fin_end, 0]
    fin_verts.append([v1, v2, v3, v4])
    # Fin 3: Vertical (+Z)
    v1 = [fin_x_start, 0, r_fin_start]; v2 = [fin_x_start, 0, r_fin_start + fin_span]
    v3 = [fin_x_end, 0, r_fin_end + fin_span * fin_taper_ratio]; v4 = [fin_x_end, 0, r_fin_end]
    fin_verts.append([v1, v2, v3, v4])
    # Fin 4: Vertical (-Z)
    v1 = [fin_x_start, 0, -r_fin_start]; v2 = [fin_x_start, 0, -(r_fin_start + fin_span)]
    v3 = [fin_x_end, 0, -(r_fin_end + fin_span * fin_taper_ratio)]; v4 = [fin_x_end, 0, -r_fin_end]
    fin_verts.append([v1, v2, v3, v4])
    
    fin_collection = Poly3DCollection(fin_verts, facecolors='darkslategrey')
    
    # == PROPELLER (THRUSTER) ==
    prop_length = 0.05
    x_prop = np.linspace(l, l + prop_length, 20)
    r_prop = np.linspace(r_final, 0.01, 20)
    
    X_prop, THETA_prop = np.meshgrid(x_prop, theta)
    R_prop, _ = np.meshgrid(r_prop, theta)
    Y_prop = R_prop * np.cos(THETA_prop)
    Z_prop = R_prop * np.sin(THETA_prop)


    # --- 3.8. Generate 2D Profile Plot (Moved and Updated) ---
    print("Generating 2D profile plot with component markers...")
    fig_2d, ax_2d = plt.subplots(figsize=(15, 6))
    
    # Plot the hull profile
    ax_2d.plot(x_nose, r_nose, 'b-', label='Nose Profile')
    ax_2d.plot(x_nose, -r_nose, 'b-')
    ax_2d.plot(x_mid, r_mid, 'g-', label='Mid-section Profile')
    ax_2d.plot(x_mid, -r_mid, 'g-')
    ax_2d.plot(x_tail, r_tail, 'r-', label='Tail Profile')
    ax_2d.plot(x_tail, -r_tail, 'r-')

    # Add lines for section breaks
    ax_2d.axvline(x=a, color='k', linestyle='--', linewidth=0.7, label=f'Nose/Mid (x={a:.3f})')
    ax_2d.axvline(x=lf, color='k', linestyle=':', linewidth=0.7, label=f'Mid/Tail (x={lf:.3f})')
    
    # == Add Component Markers ==
    
    # 1. Fins (as a shaded region)
    ax_2d.axvspan(fin_x_start, fin_x_end, color='darkslategrey', alpha=0.3, label='Fins Location')
    
    # 2. Propeller Hub (as a shaded region)
    ax_2d.axvspan(l, l + prop_length, color='black', alpha=0.3, label='Propeller Hub')

    # 3. SSS (as a thick line *on* the hull)
    sss_y = r_max + 0.005 # Slightly above the hull
    ax_2d.plot([x_sss_start, x_sss_end], [sss_y, sss_y], 'grey', linewidth=4, label='SSS Array (Top)')
    ax_2d.plot([x_sss_start, x_sss_end], [-sss_y, -sss_y], 'grey', linewidth=4, label='SSS Array (Bottom)')
    
    # 4. DVL (as a thick line *below* the hull)
    dvl_y = -r_max - 0.01 # Slightly below the hull
    ax_2d.plot([x_dvl_start, x_dvl_end], [dvl_y, dvl_y], 'orange', linewidth=4, label='DVL (Bottom)')

    # 5. Antenna Mast (as a vertical line)
    mast_y_base = r_max
    mast_y_top = r_max + mast_height
    ax_2d.plot([x_mast_base, x_mast_base], [mast_y_base, mast_y_top], 'silver', linewidth=3, label='Antenna Mast (Top)')

    # --- Formatting the 2D Plot ---
    ax_2d.set_title('AUV 2D Hull Profile and Component Placement')
    ax_2d.set_xlabel('Vehicle Length (x) (m)')
    ax_2d.set_ylabel('Vehicle Radius (r) (m)')
    ax_2d.grid(True, linestyle=':', alpha=0.6)
    
    # Adjust Y-limits to show mast and DVL marker
    ax_2d.set_ylim(-(r_max + mast_height + 0.05), r_max + mast_height + 0.05)
    # Adjust X-limits to show prop
    ax_2d.set_xlim(-0.05, l + prop_length + 0.05)
    
    # Re-order legend to be more readable
    handles, labels = ax_2d.get_legend_handles_labels()
    # A simple way to group them
    order = [0, 1, 2, 3, 4, 10, 5, 6, 7, 8, 9] 
    if len(handles) == len(order):
        ax_2d.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper right', fontsize='small')
    else:
        ax_2d.legend(loc='upper right', fontsize='small') # Fallback
        
    ax_2d.set_aspect('equal')
    plt.tight_layout()


    # --- 4. Create the 3D Plot ---
    print("Generating 3D surface model...")
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Hull
    ax.plot_surface(X_nose, Y_nose, Z_nose, color='blue', rstride=5, cstride=5, alpha=0.7)
    ax.plot_surface(X_mid, Y_mid, Z_mid, color='green', rstride=5, cstride=5, alpha=0.7)
    ax.plot_surface(X_tail, Y_tail, Z_tail, color='red', rstride=5, cstride=5, alpha=0.7)
    
    # Plot Fins and Prop
    ax.add_collection3d(fin_collection)
    ax.plot_surface(X_prop, Y_prop, Z_prop, color='black', rstride=3, cstride=3)
    
    # Plot External Components
    ax.plot_surface(X_sss1, Y_sss1, Z_sss1, color='grey')
    ax.plot_surface(X_sss2, Y_sss2, Z_sss2, color='grey')
    ax.add_collection3d(dvl_collection)
    ax.plot_surface(X_mast, Y_mast, Z_mast, color='silver')

    # --- 5. Formatting the 3D Plot ---
    
    ax.set_xlabel('X - Longitudinal Axis (m)', fontsize=12)
    ax.set_ylabel('Y - Transverse Axis (m)', fontsize=12)
    ax.set_zlabel('Z - Vertical Axis (m)', fontsize=12)
    ax.set_title('3D Model of AUV with Components', fontsize=16)

    # Set aspect ratio to be equal
    # Gather all coordinates
    all_x = np.concatenate([X_nose.flatten(), X_mid.flatten(), X_tail.flatten(), X_prop.flatten(), X_sss1.flatten(), X_sss2.flatten(), X_mast.flatten()])
    all_y = np.concatenate([Y_nose.flatten(), Y_mid.flatten(), Y_tail.flatten(), Y_prop.flatten(), Y_sss1.flatten(), Y_sss2.flatten(), Y_mast.flatten()])
    all_z = np.concatenate([Z_nose.flatten(), Z_mid.flatten(), Z_tail.flatten(), Z_prop.flatten(), Z_sss1.flatten(), Z_sss2.flatten(), Z_mast.flatten()])
    
    # Add fin and DVL vertices
    fin_verts_flat = [v_i for fin in fin_verts for v_i in fin]
    all_x = np.concatenate([all_x, [v_i[0] for v_i in fin_verts_flat], v[:,0]])
    all_y = np.concatenate([all_y, [v_i[1] for v_i in fin_verts_flat], v[:,1]])
    all_z = np.concatenate([all_z, [v_i[2] for v_i in fin_verts_flat], v[:,2]])

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    z_min, z_max = all_z.min(), all_z.max()
    
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2
    
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max()
    
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    print("Plots generated. Displaying...")
    plt.show()

if __name__ == '__main__':
    auv_params = {
        'a': 0.191,        # Nose Length (m)
        'a_offset': 0.0165, # Nose Offset (m) - NOW USED
        'b': 0.654,        # Mid-section Length (m) - not directly used
        'c': 0.541,        # Tail section Length (m) - parameter for tail equation
        'c_offset': 0.0368, # Tail Offset (m) - NOW USED
        'n': 2,            # Exponential Coefficient
        'theta_tail': 0.436,# Included angle at tail (rad) - not used
        'd': 0.191,        # Max Hull Diameter (m)
        'lf': 0.828,       # Vehicle Forward Length (m)
        'l': 1.33          # Total Vehicle Length (m)
    }
    
    # Recalculate 'c' to match l and lf
    calculated_c = auv_params['l'] - auv_params['lf']
    if abs(auv_params['c'] - calculated_c) > 0.01:
        print(f"Warning: Provided 'c' ({auv_params['c']}) does not match 'l' - 'lf' ({calculated_c:.3f}).")
        print(f"Using c = {calculated_c:.3f} for tail section calculation.")
    
    auv_params['c'] = calculated_c 

    plot_auv(**auv_params)