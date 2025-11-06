"""
This file contains the physical parameters for the AUV, based on the
"REMUS (P)"

"""

REMUS_PARAMS = {
    # Environmental
    "seawater_density": 1030,  # kg/m3

    # Geometric
    "L": 1.33,                 # m, Vehicle Length
    "D": 0.191,                # m, Vehicle max Diameter
    "Af": 0.029,               # m2, Hull Frontal Area
    "Ap": 0.226,               # m2, Hull Projected Area
    
    # Mass & Force
    "m": 30.51,                # kg, Vehicle Mass
    "W": 299,                  # N, Vehicle Weight
    "B": 306,                  # N, Vehicle Buoyancy
    
    # Center of Buoyancy (CB)
    # Origin at Vehicle nose (x=0)
    # Assuming +0.61, which is physically realistic.
    "xcb": 0.61,               # m
    "ycb": 0.0,                # m
    "zcb": 0.0,                # m
    
    # Center of Gravity (CG)
    # Origin for these values is the CB.
    "xcg_offset": 0.0,         # m (relative to xcb)
    "ycg_offset": 0.0,         # m (relative to ycb)
    "zcg_offset": 0.02,        # m (relative to zcb) - CG is 2cm *above* CB
    
    # Moments of Inertia (Origin at CB)
    "Ixx": 0.177,              # kg.m3
    "Iyy": 3.45,               # kg.m3
    "Izz": 3.45,               # kg.m3
}


# Calculate the absolute CG position
# (xcg, ycg, zcg) with origin at the vehicle nose
PARAMS_DERIVED = {
    "cb_pos": (
        REMUS_PARAMS["xcb"],
        REMUS_PARAMS["ycb"],
        REMUS_PARAMS["zcb"]
    ),
    "cg_pos": (
        REMUS_PARAMS["xcb"] + REMUS_PARAMS["xcg_offset"],
        REMUS_PARAMS["ycb"] + REMUS_PARAMS["ycg_offset"],
        REMUS_PARAMS["zcb"] + REMUS_PARAMS["zcg_offset"]
    )
}

if __name__ == '__main__':
    import json
    print("REMUS (P) Physical Parameters")
    print(json.dumps(REMUS_PARAMS, indent=2))
    print("\nDerived Positions (Origin at Nose)")
    print(json.dumps(PARAMS_DERIVED, indent=2))