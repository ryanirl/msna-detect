from msna_detect.utils.registry import * # This location becomes the new registry
from msna_detect.models import _MSNA_MODELS

add_registry("model", _MSNA_MODELS)


