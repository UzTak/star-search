"""
Functions define radii values of solar system objects
"""


# ------------------------------------------------------------------------- #
def get_radii(*args):
    """Function returns GM value of body specified by NAIF ID. 
    For body names, refer to: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html

    Args:
        naifIDs (str): tuple containing strings of naif ID to use to extract GM values. Multiple naifIDs may be passed in a single function call.

    Returns:
        (lst): lst of radii values (averaged value of x-/y-/z- radii), km

    Examples:
        >>> get_radii("399", "301")
        [398600.435436096, 4902.800066163796]
    """
    # call gm values
    pck00010 = get_radii_pck00010()
    gm_out = []
    for naifID in args:
        bdyname = "BODY" + str(naifID) + "_RADII"
        gm_out.append( sum(pck00010[bdyname]) / 3 )
    return gm_out



# ------------------------------------------------------------------------- #
def get_radii_pck00010():
    """Function returns tuple of gm values from pck00010.tpc file
    
    Args:
        None
    Returns:
        (dict): dictionary with fields defined by "BODY" + <NAIF body ID> + "_GM"", which contains a tuple of the GM value of the corresponding body
    """
    pck00010 = {
        "BODY10_RADII"  :    [ 696000.0,  696000.0,  696000.0 ],
        "BODY199_RADII" :    [ 2439.7,   2439.7,   2439.7 ],
        "BODY299_RADII" :    [ 6051.8,   6051.8,   6051.8 ],
        "BODY399_RADII" :    [ 6378.1366, 6378.1366, 6356.7519 ],
        "BODY499_RADII" :    [ 3396.19,  3396.19,  3376.20 ],
        "BODY599_RADII" :    [ 71492,    71492,    66854 ],
        "BODY699_RADII" :    [ 60268,    60268,    54364 ],
        "BODY799_RADII" :    [ 25559,    25559,    24973 ],
        "BODY899_RADII" :    [ 24764,    24764,    24341 ],
        "BODY999_RADII" :    [ 1195,     1195,     1195 ],
        
        "BODY301_RADII" :    [ 1737.4,   1737.4,   1737.4 ],
        
        "BODY401_RADII" :    [ 13.0,     11.4,     9.1 ],
        "BODY402_RADII" :    [ 7.8,      6.0,      5.1 ],
        
        "BODY501_RADII" :    [ 1829.4,   1819.4,   1815.7 ],
        "BODY502_RADII" :    [ 1562.6,   1560.3,   1559.5 ],
        "BODY503_RADII" :    [ 2631.2,   2631.2,   2631.2 ],
        "BODY504_RADII" :    [ 2410.3,   2410.3,   2410.3 ],
        "BODY505_RADII" :    [ 125,      73,       64 ],
        
        "BODY601_RADII" :    [ 207.8,    196.7,    190.6 ],
        "BODY602_RADII" :    [ 256.6,    251.4,    248.3 ],
        "BODY603_RADII" :    [ 538.4,    528.3,    526.3 ],
        "BODY604_RADII" :    [ 563.4,    561.3,    559.6 ],
        "BODY605_RADII" :    [ 765.0,    763.1,    762.4 ],
        "BODY606_RADII" :    [ 2575.15,  2574.78,  2574.47 ],
        "BODY607_RADII" :    [ 180.1,    133.0,    102.7 ],
        "BODY608_RADII" :    [ 745.7,    745.7,    712.1 ],
        "BODY609_RADII" :    [ 109.4,    108.5,    101.8 ],
        "BODY610_RADII" :    [ 101.5,    92.5,     76.3 ],
        "BODY611_RADII" :    [ 64.9,     57.0,     53.1 ],
        "BODY612_RADII" :    [ 21.7,     19.1,     13.0 ],
        "BODY613_RADII" :    [ 16.3,     11.8,     10.0 ],
        "BODY614_RADII" :    [ 15.1,     11.5,     7.0 ],
        "BODY615_RADII" :    [ 20.4,     17.7,     9.4 ],
        "BODY616_RADII" :    [ 67.8,     39.7,     29.7 ],
        "BODY617_RADII" :    [ 52.0,     40.5,     32.0 ],
        "BODY618_RADII" :    [ 17.2,     15.7,     10.4 ],
        
        # csc 2026 asteroid 
        "BODY1001673_RADII" :  [1, 1, 1],
}

    return pck00010