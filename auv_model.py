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
        pass

def plot_auv(a, a_offset, c, c_offset, n, d, lf, l, cb_pos, cg_pos):
    """
    Generates and plots a 3D model and 2D profile of a torpedo-shaped AUV.
    Args:
        ... (geometric args) ...
        cb_pos (tuple): (x, y, z) for Center of Buoyancy
        cg_pos (tuple): (x, y, z) for Center of Gravity
    """
    
    r_max = d / 2
    num_x_points = 100
    num_theta_points = 80
    theta = np.linspace(0, 2 * np.pi, num_theta_points)
    z_offset = 0.001 


    # NOSE SECTION
    x_nose = np.linspace(0, a, num_x_points)
    r_nose = r_max - (r_max - a_offset) * (1 - x_nose / a)**n
    
    X_nose, THETA_nose = np.meshgrid(x_nose, theta)
    R_nose, _ = np.meshgrid(r_nose, theta)
    Y_nose = R_nose * np.cos(THETA_nose)
    Z_nose = R_nose * np.sin(THETA_nose)

    # MID-SECTION (Cylinder)
    mid_section_length = lf - a
    num_x_mid_points = max(2, int(num_x_points * (mid_section_length / a)))
    x_mid = np.linspace(a, lf, num_x_mid_points)
    r_mid = np.full_like(x_mid, r_max)
    
    X_mid, THETA_mid = np.meshgrid(x_mid, theta)
    R_mid, _ = np.meshgrid(r_mid, theta)
    Y_mid = R_mid * np.cos(THETA_mid)
    Z_mid = R_mid * np.sin(THETA_mid)

    # TAIL SECTION (Power Series Curve of Revolution)
    num_x_tail_points = max(2, int(num_x_points * (c / a)))
    x_tail = np.linspace(lf, l, num_x_tail_points)
    
    x_norm_tail = (x_tail - lf) / c
    r_tail = c_offset + (r_max - c_offset) * (1 - x_norm_tail**n)
    
    X_tail, THETA_tail = np.meshgrid(x_tail, theta)
    R_tail, _ = np.meshgrid(r_tail, theta)
    Y_tail = R_tail * np.cos(THETA_tail)
    Z_tail = R_tail * np.sin(THETA_tail)
    
    r_final = c_offset # Final radius at the tail (propeller hub)

    # Side-Scan Sonar (SSS) Patches 
    sss_len = 0.3; sss_width_angle = 0.1
    x_sss_start = a + (mid_section_length - sss_len) / 2
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
    x_dvl_start = lf - dvl_len - 0.05
    x_dvl_end = x_dvl_start + dvl_len
    y_dvl_half = dvl_width / 2; z_dvl_top = -r_max; z_dvl_bottom = z_dvl_top - dvl_height
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
    dvl_collection = Poly3DCollection(dvl_faces, facecolors='orange')

    # Antenna Mast
    mast_height = 0.08; mast_radius = 0.01; x_mast_base = lf - 0.1
    theta_mast = np.linspace(0, 2 * np.pi, 20)
    z_mast = np.linspace(r_max, r_max + mast_height, 2)
    TH_mast, Z_mast = np.meshgrid(theta_mast, z_mast)
    X_mast = x_mast_base + mast_radius * np.cos(TH_mast)
    Y_mast = 0 + mast_radius * np.sin(TH_mast)
        
    fin_length = 0.12; fin_span = 0.1; fin_taper_ratio = 0.8
    fin_x_end = l - 0.025 
    fin_x_start = fin_x_end - fin_length
    
    x_norm_fin_start = (fin_x_start - lf) / c
    r_fin_start = c_offset + (r_max - c_offset) * (1 - x_norm_fin_start**n)
    x_norm_fin_end = (fin_x_end - lf) / c
    r_fin_end = c_offset + (r_max - c_offset) * (1 - x_norm_fin_end**n)
    
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
    
    fin_collection = Poly3DCollection(fin_verts, facecolors='darkslategrey')
    
    # 4-Bladed Propeller
    prop_tip_radius = fin_span * 1.0 
    prop_hub_radius = r_final
    prop_pitch = 0.1
    prop_chord_angle = np.pi / 8
    prop_x_pos = l
    
    prop_blades = []
    r_blade = np.linspace(prop_hub_radius, prop_tip_radius, 8)
    th_blade_base = np.linspace(-prop_chord_angle/2, prop_chord_angle/2, 5)
    
    for i in range(4): # 4 blades
        base_angle = i * (np.pi / 2)
        R_blade, TH_blade = np.meshgrid(r_blade, th_blade_base)
        Y_blade = R_blade * np.cos(TH_blade + base_angle)
        Z_blade = R_blade * np.sin(TH_blade + base_angle)
        X_blade = prop_x_pos + (R_blade * prop_pitch) * np.sin(TH_blade)
        prop_blades.append((X_blade, Y_blade, Z_blade))

    # Protective Cage (Shroud)
    cage_radius = prop_tip_radius + 0.015
    cage_length = 0.08
    cage_x_start = l - 0.02
    cage_x_end = cage_x_start + cage_length
    
    x_cage = np.linspace(cage_x_start, cage_x_end, 10)
    theta_cage = np.linspace(0, 2 * np.pi, 40)
    
    X_cage, TH_cage = np.meshgrid(x_cage, theta_cage)
    Y_cage = cage_radius * np.cos(TH_cage)
    Z_cage = cage_radius * np.sin(TH_cage)
    
    # Define Physics Markers
    marker_radius = 0.02
    X_cb, Y_cb, Z_cb = create_sphere_marker(cb_pos, marker_radius)
    X_cg, Y_cg, Z_cg = create_sphere_marker(cg_pos, marker_radius)


    # Generate 2D Profile Plot
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
    
    # Add Component Markers
    ax_2d.axvspan(fin_x_start, fin_x_end, color='darkslategrey', alpha=0.3, label='Fins Location')
    ax_2d.axvspan(cage_x_start, cage_x_end, color='black', alpha=0.3, label='Propeller & Shroud')
    sss_y = r_max + 0.005
    ax_2d.plot([x_sss_start, x_sss_end], [sss_y, sss_y], 'grey', linewidth=4, label='SSS Array (Top)')
    ax_2d.plot([x_sss_start, x_sss_end], [-sss_y, -sss_y], 'grey', linewidth=4)
    dvl_y = -r_max - 0.01
    ax_2d.plot([x_dvl_start, x_dvl_end], [dvl_y, dvl_y], 'orange', linewidth=4, label='DVL (Bottom)')
    mast_y_base = r_max
    mast_y_top = r_max + mast_height
    ax_2d.plot([x_mast_base, x_mast_base], [mast_y_base, mast_y_top], 'silver', linewidth=3, label='Antenna Mast (Top)')

    # Add Physics Markers
    ax_2d.plot(cb_pos[0], cb_pos[2], 'bo', markersize=10, label='CB (Buoyancy)')
    ax_2d.plot(cg_pos[0], cg_pos[2], 'rX', markersize=10, label='CG (Gravity)')

    ax_2d.set_title('AUV 2D Hull Profile and Component Placement')
    ax_2d.set_xlabel('Vehicle Length (x) (m)')
    ax_2d.set_ylabel('Vehicle Radius (r) (m)')
    ax_2d.grid(True, linestyle=':', alpha=0.6)
    
    ax_2d.set_ylim(-(r_max + mast_height + 0.05), r_max + mast_height + 0.05)
    ax_2d.set_xlim(-0.05, cage_x_end + 0.05) 
    
    ax_2d.legend(loc='upper right', fontsize='small')
    ax_2d.set_aspect('equal')
    plt.tight_layout()


    # Create the 3D Plot
    print("Generating 3D surface model...")
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Hull
    ax.plot_surface(X_nose, Y_nose, Z_nose, color='blue', rstride=5, cstride=5, alpha=0.7)
    ax.plot_surface(X_mid, Y_mid, Z_mid, color='green', rstride=5, cstride=5, alpha=0.7)
    ax.plot_surface(X_tail, Y_tail, Z_tail, color='red', rstride=5, cstride=5, alpha=0.7)
    
    # Plot Fins
    ax.add_collection3d(fin_collection)
    
    # Plot External Components
    ax.plot_surface(X_sss1, Y_sss1, Z_sss1, color='grey')
    ax.plot_surface(X_sss2, Y_sss2, Z_sss2, color='grey')
    ax.add_collection3d(dvl_collection)
    ax.plot_surface(X_mast, Y_mast, Z_mast, color='silver')
    
    # Plot Propeller Blades
    for X_b, Y_b, Z_b in prop_blades:
        ax.plot_surface(X_b, Y_b, Z_b, color='black')
        
    # Plot Propeller Cage
    ax.plot_surface(X_cage, Y_cage, Z_cage, color='grey', alpha=0.4)
    
    # Plot Physics Markers (NEW)
    # Center of Buoyancy (Blue Sphere)
    ax.plot_surface(X_cb, Y_cb, Z_cb, color='blue', alpha=1.0)
    # Center of Gravity (Red Sphere)
    ax.plot_surface(X_cg, Y_cg, Z_cg, color='red', alpha=1.0)
    
    ax.set_xlabel('X - Longitudinal Axis (m)', fontsize=12)
    ax.set_ylabel('Y - Transverse Axis (m)', fontsize=12)
    ax.set_zlabel('Z - Vertical Axis (m)', fontsize=12)
    ax.set_title('3D Model of AUV with Components and Physics Markers', fontsize=16)

    # Set aspect ratio to be equal
    all_x = np.concatenate([X_nose.flatten(), X_mid.flatten(), X_tail.flatten(), X_sss1.flatten(), X_sss2.flatten(), X_mast.flatten(), X_cage.flatten(), X_cb.flatten(), X_cg.flatten()])
    all_y = np.concatenate([Y_nose.flatten(), Y_mid.flatten(), Y_tail.flatten(), Y_sss1.flatten(), Y_sss2.flatten(), Y_mast.flatten(), Y_cage.flatten(), Y_cb.flatten(), Y_cg.flatten()])
    all_z = np.concatenate([Z_nose.flatten(), Z_mid.flatten(), Z_tail.flatten(), Z_sss1.flatten(), Z_sss2.flatten(), Z_mast.flatten(), Z_cage.flatten(), Z_cb.flatten(), Z_cg.flatten()])
    
    fin_verts_flat = [v_i for fin in fin_verts for v_i in fin]
    all_x = np.concatenate([all_x, [v_i[0] for v_i in fin_verts_flat], v[:,0]])
    all_y = np.concatenate([all_y, [v_i[1] for v_i in fin_verts_flat], v[:,1]])
    all_z = np.concatenate([all_z, [v_i[2] for v_i in fin_verts_flat], v[:,2]])
    
    for X_b, Y_b, Z_b in prop_blades:
        all_x = np.concatenate([all_x, X_b.flatten()])
        all_y = np.concatenate([all_y, Y_b.flatten()])
        all_z = np.concatenate([all_z, Z_b.flatten()])

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

    plot_auv(
        c=calculated_c,
        cb_pos=cb_pos,
        cg_pos=cg_pos,
        **auv_geo
    )