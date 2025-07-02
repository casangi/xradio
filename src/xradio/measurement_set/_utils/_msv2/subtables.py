subt_rename_ids = {
    "ANTENNA": {"row": "antenna_id", "dim_1": "xyz"},
    "FEED": {"dim_1": "xyz", "dim_2": "receptor", "dim_3": "receptor2"},
    "FIELD": {"row": "field_id", "dim_1": "poly_id", "dim_2": "ra/dec"},
    "FREQ_OFFSET": {"antenna1": "antenna1_id", "antenna2": "antenna2_id"},
    "OBSERVATION": {"row": "observation_id", "dim_1": "start/end"},
    "POINTING": {"dim_1": "n_polynomial", "dim_3": "dir"},
    "POLARIZATION": {"row": "pol_setup_id", "dim_2": "product_id"},
    "PROCESSOR": {"row": "processor_id"},
    "SPECTRAL_WINDOW": {"row": "spectral_window_id", "dim_1": "chan"},
    "SOURCE": {"dim_1": "ra/dec", "dim_2": "line"},
    "STATE": {"row": "state_id"},
    "SYSCAL": {"dim_1": "frequency", "dim_2": "receptor"},
    # Would make sense for non-std "WS_NX_STATION_POSITION"
    "WEATHER": {"dim_1": "xyz"},
}
