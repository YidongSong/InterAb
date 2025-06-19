"""Import this script to avoid errors when '@profile' is used for performance profiling.

Example:
> from tfold.utils.prof_utils import *
"""

try:
    profile  # pylint: disable=used-before-assignment
except NameError:
    def profile(func):
        """Directly execute the function without performance profiling."""
        return func
