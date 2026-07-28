"""
Functions define semi-major axis values of solar system objects
"""


# ------------------------------------------------------------------------- #

def get_semiMajorAxes(*args):
    """Function returns semi-major axis value of body specified by NAIF ID. 
    For body names, refer to: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html

    Args:
        naifIDs (str): tuple containing strings of naif ID to use to extract GM values. Multiple naifIDs may be passed in a single function call.

    Returns:
        (lst): lst of semi-major axis values, in km

    Examples:
        >>> get_semiMajorAxes("399", "301")
        [149600000.0, 384400.0]
    """
    # call gm values
    de431a = get_semiMajorAxes_dict()
    a_out = []
    for naifID in args:
        bdyname = "BODY" + str(naifID) + "_semiMajorAxis"
        a_out.append( de431a[bdyname] )
    return a_out



# ------------------------------------------------------------------------- #
def get_semiMajorAxes_dict():
    """get_semiMajorAxes_dict returns tuple of gm values from de431 spice kernel
    
    Args:
        None
    Returns:
        (dict): dictionary with fields defined by "BODY" + <NAIF body ID> + "_semiMajorAxis"", which contains a tuple of the GM value of the corresponding body
    """
    de431 = {
        "BODY1_semiMajorAxis"       : 2.2031780000000021E+04 ,
        "BODY2_semiMajorAxis"       : 3.2485859200000006E+05 ,
        "BODY3_semiMajorAxis"       : 4.0350323550225981E+05 ,
        "BODY4_semiMajorAxis"       : 4.2828375214000022E+04 ,
        "BODY5_semiMajorAxis"       : 1.2671276480000021E+08 ,
        "BODY6_semiMajorAxis"       : 3.7940585200000003E+07 ,
        "BODY7_semiMajorAxis"       : 5.7945486000000080E+06 ,
        "BODY8_semiMajorAxis"       : 6.8365271005800236E+06 ,
        "BODY9_semiMajorAxis"       : 9.7700000000000068E+02 ,
        "BODY10_semiMajorAxis"      : 1.3271244004193938E+11 ,

        "BODY199_semiMajorAxis"     :  57.91e6  ,
        "BODY299_semiMajorAxis"     : 108.21e6  ,
        "BODY399_semiMajorAxis"     : 149.60e6  ,
        "BODY499_semiMajorAxis"     : 227.92e6  ,
        "BODY599_semiMajorAxis"     : 778.57e6  ,
        "BODY699_semiMajorAxis"     : 1433.53e6 ,
        "BODY799_semiMajorAxis"     : 2872.46e6 ,
        "BODY899_semiMajorAxis"     : 4495.06e6 ,
        "BODY999_semiMajorAxis"     : 5906.38e6 ,

        "BODY301_semiMajorAxis"     : 0.3844e6 ,

        "BODY401_semiMajorAxis"     : 9378.0 ,
        "BODY402_semiMajorAxis"     : 23459.0 ,

        "BODY501_semiMajorAxis"     : 431.8e3 ,
        "BODY502_semiMajorAxis"     : 671.1e3 ,
        "BODY503_semiMajorAxis"     : 1070.4e3 ,
        "BODY504_semiMajorAxis"     : 1882.7e3 ,
        "BODY505_semiMajorAxis"     : 181.4e3 ,

        "BODY601_semiMajorAxis"     : 185.52e3 ,
        "BODY602_semiMajorAxis"     : 238.02e3 ,
        "BODY603_semiMajorAxis"     : 294.64e3 ,
        "BODY604_semiMajorAxis"     : 377.60e3 ,
        "BODY605_semiMajorAxis"     : 527.04e3 ,
        "BODY606_semiMajorAxis"     : 1221.83e3 ,
        "BODY607_semiMajorAxis"     : 1481.1e3 ,
        "BODY608_semiMajorAxis"     : 3561.3e3 ,

        "BODY701_semiMajorAxis"     : 190.90e3 ,
        "BODY702_semiMajorAxis"     : 266.00e3 ,
        "BODY703_semiMajorAxis"     : 436.39e3 ,
        "BODY704_semiMajorAxis"     : 583.50e3 ,
        "BODY705_semiMajorAxis"     : 129.90e3 ,

        "BODY801_semiMajorAxis"     : 354.76e3,

        "BODY901_semiMajorAxis"     : 19596 ,
        "BODY902_semiMajorAxis"     : 48690 ,
        "BODY903_semiMajorAxis"     : 64740 ,
        "BODY904_semiMajorAxis"     : 57780,
        
    }
    return de431