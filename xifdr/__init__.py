from . import boosting
from . import fdr
import setuptools_scm

def get_version():
    # Try to get version from version file
    try:
        from ._version import __version__
        return __version__
    except:
        pass
    # Try to get version from workdir
    try:
        return setuptools_scm.get_version()
    except:
        pass
    # No version found
    return "unkown"

__version__ = get_version()