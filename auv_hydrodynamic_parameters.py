"""
This file contains all the hydrodynamic parameters (added mass, damping, etc.)
required by the 6-DOF model equations.

The values here are **PLACEHOLDERS** or estimates for a REMUS-like AUV.

All units are SI units (kg, m, s, N, etc.).
"""

import numpy as np

HYDRO_PARAMS = {
    'Xu_dot': -5.0,     # kg, Added mass in surge
    'Yv_dot': -20.0,    # kg, Added mass in sway
    'Zw_dot': -20.0,    # kg, Added mass in heave
    'Kp_dot': -1.0,     # kg*m^2, Added mass in roll
    'Mq_dot': -10.0,    # kg*m^2, Added mass in pitch
    'Nr_dot': -10.0,    # kg*m^2, Added mass in yaw
    
    # Cross-terms for added mass (often non-zero for asymmetric bodies)
    'Y_r_dot': -2.0,     # kg*m, Added mass coupling sway-yaw
    'Z_q_dot': -2.0,     # kg*m, Added mass coupling heave-pitch
    'M_w_dot': -2.0,     # kg*m, Added mass coupling pitch-heave
    'N_v_dot': -2.0,     # kg*m, Added mass coupling yaw-sway
}

HYDRO_PARAMS.update({
    'Xu': -20.0,     # N/(m/s), Linear damping in surge
    'Yv': -50.0,     # N/(m/s), Linear damping in sway
    'Zw': -50.0,     # N/(m/s), Linear damping in heave
    'Kp': -5.0,      # N*m/(rad/s), Linear damping in roll
    'Mq': -20.0,     # N*m/(rad/s), Linear damping in pitch
    'Nr': -20.0,     # N*m/(rad/s), Linear damping in yaw

    # Linear damping cross-terms (often small or zero)
    'Yp': 0.0,
    'Yr': -5.0,    # N/(rad/s)
    'Zq': -5.0,    # N/(rad/s)
    'Kv': 0.0,
    'Kr': 0.0,
    'Mw': -5.0,    # N*m/(m/s)
    'Nv': -5.0,    # N*m/(m/s)
    'Np': 0.0,
})

HYDRO_PARAMS.update({
    # Dn1 (Diagonal, |v|v terms)
    'X_u|u|': -30.0,    # N/(m/s)^2
    'Y_v|v|': -100.0,   # N/(m/s)^2
    'Z_w|w|': -100.0,   # N/(m/s)^2
    'K_p|p|': -10.0,    # N*m/(rad/s)^2
    'M_q|q|': -50.0,    # N*m/(rad/s)^2
    'N_r|r|': -50.0,    # N*m/(rad/s)^2

    # Cross-terms (often zero if diagonal is used)
    'Y_r|r|': -2.0,
    'Z_q|q|': -2.0,
    'M_w|w|': -2.0,
    'N_v|v|': -2.0,
    
    # Dn2 (Lift forces, cross-flow drag)
    'Yuv': 0.0,
    'Yur': -50.0,   # N/(m/s * rad/s)
    'Zuw': -50.0,   # N/(m/s * m/s)
    'Zuq': -50.0,   # N/(m/s * rad/s)
    'Kupu': 0.0,  # Assuming 'Kupu' from text
    'Muw': -50.0,   # N*m/(m/s * m/s)
    'Muq': -50.0,   # N*m/(m/s * rad/s)
    'Nuv': -50.0,   # N*m/(m/s * m/s)
    'Nur': -50.0,   # N*m/(m/s * rad/s)
})

HYDRO_PARAMS.update({
    'Yuu_delta_r': -15.0,   # N / ( (m/s)^2 * rad )
    'Zuu_delta_s': 15.0,    # N / ( (m/s)^2 * rad )
    'Muu_delta_s': 10.0,    # N*m / ( (m/s)^2 * rad )
    'Nuu_delta_r': -10.0,   # N*m / ( (m/s)^2 * rad )
    'Kroll_delta_roll': -10.0 # N*m / rad (if roll fins are used)
})

HYDRO_PARAMS.update({
    # Thrust (X-force)
    'prop_a1': 0.005,
    'prop_a2': 0.0,
    'prop_a3': 0.0,
    # Torque (K-moment)
    'prop_b1': 0.001,
    'prop_b2': 0.0,
    'prop_b3': 0.0,
})


if __name__ == '__main__':
    """ Utility to print all defined parameters """
    import json
    print(f"Loaded {len(HYDRO_PARAMS)} hydrodynamic parameters.")
    print(json.dumps(HYDRO_PARAMS, indent=4))
