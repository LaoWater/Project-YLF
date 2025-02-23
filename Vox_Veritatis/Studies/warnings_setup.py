import sys
import os
from contextlib import contextmanager


# Context manager to suppress print output
@contextmanager
def suppress_stdout():
    # Redirect stdout to /dev/null to suppress print statements
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
