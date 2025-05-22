import numpy as np
import numpy.random as random

from typing import Tuple


class Compose:
    def __init__(self, layers): 
        self.layers = layers

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for layer in self.layers:
            x, y = layer(x, y)

        return x, y


class HorizontalFlip:
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            x = np.ascontiguousarray(x[:, ::-1])
            y = np.ascontiguousarray(y[:, ::-1])

        return x, y


class GaussianNoise:
    def __init__(self, p: float, sigma_l: float = 0.1, sigma_r: float = 0.1) -> None:
        self.p = p
        self.sigma_l = sigma_l
        self.sigma_r = sigma_r

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            sigma = np.random.uniform(self.sigma_l, self.sigma_r)
            x = x + random.normal(0, sigma, size = x.shape).astype(np.float32)

        return x, y


