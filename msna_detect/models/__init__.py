from .unet import Unet1d

from typing import Dict
from typing import Type

import torch.nn as nn

_MSNA_MODELS: Dict[str, Type[nn.Module]] = {
    "unet1d": Unet1d
}


def get_model(model: str, **params) -> nn.Module:
    if model.lower() not in _MSNA_MODELS:
        raise ValueError(
            f"The model '{model}' is not supported. Options are "
            f"{_MSNA_MODELS.keys()}."
        )

    return _MSNA_MODELS[model](**params)


