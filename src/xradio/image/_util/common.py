def __get_xds_dim_order(has_sph:bool) -> list:
    dimorder = ['time', 'pol', 'freq']
    dir_lin = ['l', 'm'] if has_sph else ['u', 'v']
    dimorder.extend(dir_lin)
    return dimorder


