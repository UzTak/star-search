"""
Functions define GM values of solar system objects
"""


# ------------------------------------------------------------------------- #
def get_gm(*args):
    """Function returns GM value of body specified by NAIF ID. 
    For body names, refer to: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html

    Args:
        naifIDs (str): tuple containing strings of naif ID to use to extract GM values. Multiple naifIDs may be passed in a single function call.

    Returns:
        (lst): lst of GM values, km^3/s^2

    Examples:
        >>> get_gm("399", "301")
        [398600.435436096, 4902.800066163796]
    """
    # call gm values
    # de_dict = get_gm_de431()
    de_dict = get_gm_de440()
    
    gm_out = []
    for naifID in args:
        bdyname = "BODY" + str(naifID) + "_GM"
        gm_out.append( de_dict[bdyname])
    return gm_out



# ------------------------------------------------------------------------- #
def get_gm_de431():
    """Function returns tuple of gm values from de431 spice kernel
    
    Args:
        None
    Returns:
        (dict): dictionary with fields defined by "BODY" + <NAIF body ID> + "_GM"", which contains a tuple of the GM value of the corresponding body
    """
    de431 = {
        "BODY1_GM"       : 2.2031780000000021E+04 ,
        "BODY2_GM"       : 3.2485859200000006E+05 ,
        "BODY3_GM"       : 4.0350323550225981E+05 ,
        "BODY4_GM"       : 4.2828375214000022E+04 ,
        "BODY5_GM"       : 1.2671276480000021E+08 ,
        "BODY6_GM"       : 3.7940585200000003E+07 ,
        "BODY7_GM"       : 5.7945486000000080E+06 ,
        "BODY8_GM"       : 6.8365271005800236E+06 ,
        "BODY9_GM"       : 9.7700000000000068E+02 ,
        "BODY10_GM"      : 1.3271244004193938E+11 ,

        "BODY199_GM"     : 2.2031780000000021E+04 ,
        "BODY299_GM"     : 3.2485859200000006E+05 ,
        "BODY399_GM"     : 3.9860043543609598E+05 ,
        "BODY499_GM"     : 4.282837362069909E+04  ,
        "BODY599_GM"     : 1.266865349218008E+08  ,
        "BODY699_GM"     : 3.793120749865224E+07  ,
        "BODY799_GM"     : 5.793951322279009E+06  ,
        "BODY899_GM"     : 6.835099502439672E+06  ,
        "BODY999_GM"     : 8.696138177608748E+02  ,

        "BODY301_GM"     : 4.9028000661637961E+03 ,

        "BODY401_GM"     : 7.087546066894452E-04 ,
        "BODY402_GM"     : 9.615569648120313E-05 ,

        "BODY501_GM"     : 5.959916033410404E+03 ,
        "BODY502_GM"     : 3.202738774922892E+03 ,
        "BODY503_GM"     : 9.887834453334144E+03 ,
        "BODY504_GM"     : 7.179289361397270E+03 ,
        "BODY505_GM"     : 1.378480571202615E-01 ,

        "BODY601_GM"     : 2.503522884661795E+00 ,
        "BODY602_GM"     : 7.211292085479989E+00 ,
        "BODY603_GM"     : 4.121117207701302E+01 ,
        "BODY604_GM"     : 7.311635322923193E+01 ,
        "BODY605_GM"     : 1.539422045545342E+02 ,
        "BODY606_GM"     : 8.978138845307376E+03 ,
        "BODY607_GM"     : 3.718791714191668E-01 ,
        "BODY608_GM"     : 1.205134781724041E+02 ,
        "BODY609_GM"     : 5.531110414633374E-01 ,
        "BODY610_GM"     : 1.266231296945636E-01 ,
        "BODY611_GM"     : 3.513977490568457E-02 ,
        "BODY615_GM"     : 3.759718886965353E-04 ,
        "BODY616_GM"     : 1.066368426666134E-02 ,
        "BODY617_GM"     : 9.103768311054300E-03 ,

        "BODY701_GM"     : 8.346344431770477E+01 ,
        "BODY702_GM"     : 8.509338094489388E+01 ,
        "BODY703_GM"     : 2.269437003741248E+02 ,
        "BODY704_GM"     : 2.053234302535623E+02 ,
        "BODY705_GM"     : 4.319516899232100E+00 ,

        "BODY801_GM"     : 1.427598140725034E+03 ,

        "BODY901_GM"     : 1.058799888601881E+02 ,
        "BODY902_GM"     : 3.048175648169760E-03 ,
        "BODY903_GM"     : 3.211039206155255E-03 ,
        "BODY904_GM"     : 1.110040850536676E-03 ,
        
        # csc 2026 asteroid 
        "BODY1001673_GM" : 1.0E-03 ,
    }
    return de431


def get_gm_de440():
    """Function returns tuple of gm values from de431 spice kernel
    
    Args:
        None
    Returns:
        (dict): dictionary with fields defined by "BODY" + <NAIF body ID> + "_GM"", which contains a tuple of the GM value of the corresponding body
    """
    
    de440 = {
        "BODY1_GM"     : 2.2031868551400003E+04 ,
        "BODY2_GM"     : 3.2485859200000000E+05 ,
        "BODY3_GM"     : 4.0350323562548019E+05 ,
        "BODY4_GM"     : 4.2828375815756102E+04 ,
        "BODY5_GM"     : 1.2671276409999998E+08 ,
        "BODY6_GM"     : 3.7940584841799997E+07 ,
        "BODY7_GM"     : 5.7945563999999985E+06 ,
        "BODY8_GM"     : 6.8365271005803989E+06 ,
        "BODY9_GM"     : 9.7550000000000000E+02 ,
        "BODY10_GM"    : 1.3271244004127942E+11 ,

        "BODY199_GM"   : 2.2031868551400003E+04 ,
        "BODY299_GM"   : 3.2485859200000000E+05 ,
        "BODY301_GM"   : 4.9028001184575496E+03 ,
        "BODY399_GM"   : 3.9860043550702266E+05 ,

        "BODY499_GM"   : 4.282837362069909E+04 ,
        "BODY599_GM"   : 1.266865319003704E+08 ,
        "BODY699_GM"   : 3.793120623436167E+07 ,
        "BODY799_GM"   : 5.793951256527211E+06 ,
        "BODY899_GM"   : 6.835103145462294E+06 ,
        "BODY999_GM"   : 8.696138177608748E+02 ,

        "BODY401_GM"   : 7.087546066894452E-04 ,
        "BODY402_GM"   : 9.615569648120313E-05 ,

        "BODY501_GM"   : 5.959915466180539E+03 ,
        "BODY502_GM"   : 3.202712099607295E+03 ,
        "BODY503_GM"   : 9.887832752719638E+03 ,
        "BODY504_GM"   : 7.179283402579837E+03 ,
        "BODY505_GM"   : 1.645634534798259E-01 ,
        "BODY506_GM"   : 1.515524299611265E-01 ,
        "BODY514_GM"   : 3.014800000000000E-02 ,
        "BODY515_GM"   : 1.390000000000000E-04 ,
        "BODY516_GM"   : 2.501000000000000E-03 ,

        "BODY601_GM"   : 2.503488768152587E+00 ,
        "BODY602_GM"   : 7.210366688598896E+00 ,
        "BODY603_GM"   : 4.121352885489587E+01 ,
        "BODY604_GM"   : 7.311607172482067E+01 ,
        "BODY605_GM"   : 1.539417519146563E+02 ,
        "BODY606_GM"   : 8.978137095521046E+03 ,
        "BODY607_GM"   : 3.704913747932265E-01 ,
        "BODY608_GM"   : 1.205151060137642E+02 ,
        "BODY609_GM"   : 5.547860052791678E-01 ,
        "BODY610_GM"   : 1.265765099012197E-01 ,
        "BODY611_GM"   : 3.512333288208074E-02 ,
        "BODY612_GM"   : 4.757419551776972E-04 ,
        "BODY615_GM"   : 3.718871247516475E-04 ,
        "BODY616_GM"   : 1.075208001007610E-02 ,
        "BODY617_GM"   : 9.290325122028795E-03 ,

        "BODY701_GM"   : 8.346344431770477E+01 ,
        "BODY702_GM"   : 8.509338094489388E+01 ,
        "BODY703_GM"   : 2.269437003741248E+02 ,
        "BODY704_GM"   : 2.053234302535623E+02 ,
        "BODY705_GM"   : 4.319516899232100E+00 ,

        "BODY801_GM"   : 1.428495462910464E+03 ,
        "BODY803_GM"   : 8.530281246540886E-03 ,
        "BODY804_GM"   : 2.358873197992170E-02 ,
        "BODY805_GM"   : 1.167318403814998E-01 ,
        "BODY806_GM"   : 1.898985039060690E-01 ,
        "BODY807_GM"   : 2.548437405693583E-01 ,
        "BODY808_GM"   : 2.583422379120727E+00 ,

        "BODY901_GM"   : 1.058799888601881E+02 ,
        "BODY902_GM"   : 3.048175648169760E-03 ,
        "BODY903_GM"   : 3.211039206155255E-03 ,
        "BODY904_GM"   : 1.110040850536676E-03 ,
        "BODY905_GM"   : 0.000000000000000E+00 ,
        
        # csc 2026 asteroid 
        "BODY1001673_GM" : 1.0E-03 ,
    }
    
    return de440