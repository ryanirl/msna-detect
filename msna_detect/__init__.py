# MSNA Burst Detection
#
# A deep learning framework for automated detection of bursts in Muscle
# Sympathetic Nerve Activity (MSNA) signals.
#
__version__ = "0.1.0"
__author__ = "Ryan 'RyanIRL' Peters"
__license__ = "MIT"

# Import core modules
import msna_detect.utils
import msna_detect.model
import msna_detect.registry  # Populate the registries

# The main user facing model with support for training and inference
from msna_detect.model import MsnaModel

__all__ = [
    "MsnaModel",
    "__version__",
    "__author__",
    "__license__",
]


