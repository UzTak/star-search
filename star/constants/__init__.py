# import functions that defines solar system constants 

from ._get_gm import get_gm, get_gm_de431
from ._get_sma import get_semiMajorAxes, get_semiMajorAxes_dict
from ._get_radii import get_radii, get_radii_pck00010

# Shared constants for numerical foundations.

SECONDS_PER_DAY = 86400.0
J2000_JD_TDB = 2451545.0

# Default heliocentric inertial setup.
DEFAULT_FRAME = "ECLIPJ2000"
DEFAULT_OBSERVER = "SUN"