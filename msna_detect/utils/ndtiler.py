################################################################################
# Lightweight package to support iterative tiling over slices of ND-matrices. 
# Currently, `ndtiler` supports two tiling strategies: A static (default) and
# minimally dynamic tiling strategy. 
# 
# It's important to note that not all combinations of parameters work as
# `ndtiler` requires even sampling of the input shape (`size`) in the static
# tiling strategy. For these cases, you would need to first use `get_overflow()` 
# to first determine how much one needs to adjust their input matrix (with
# padding or resizing for example) to match the tiling size and overlap
# provided. The dynamic strategy on the other hand only requires that the tile
# size is larger than the provided matrix size (`size`).
# 
# For the reasons mentioned above, `ndtiler` is a lower-level library that 
# developers should use to build higher-level functionality on top of. 
#
# Author: Ryan 'RyanIRL' Peters
# Email: RyanIRL@icloud.com
# License: MIT
#
################################################################################

from typing import Iterator
from typing import Tuple
from typing import List

from math import ceil


def tile_nd(
    size: Tuple[int, ...], tile_size: Tuple[int, ...], overlap: Tuple[int, ...]
) -> Iterator[Tuple[Tuple[int, int], ...]]:
    """
    Generate the coordinates for tiling an N-dimensional array. Example usage
    can be found in `ryanirl/ndtiler/examples/*`.

    Args:
        size (Tuple[int, ...]): The size of the array.
        tile_size (Tuple[int, ...]): The size of each tile.
        overlap (Tuple[int, ...]): The overlap between adjacent tiles.

    Yields:
        Iterator[Tuple[Tuple[int, int], ...]]: The coordinates of each tile.
    """
    _check_positive_stride(size, tile_size, overlap)
    _check_tile_size(size, tile_size, overlap)
    _check_size_large(size, tile_size, overlap)
    _check_even_tiling(size, tile_size, overlap)
    _check_same_dimensions(size, tile_size, overlap)

    stride = get_stride(tile_size, overlap)
    tile_count = get_tile_count(size, tile_size, overlap)
    for index in _generate_inds(0, [], tile_count):
        tiles = []
        for i, l in enumerate(index):
            # Convert the tile index to the proper shape, then add it to tiles.
            ls = l * stride[i]
            tiles.append((ls, ls + tile_size[i]))

        yield tuple(tiles)


def dynamic_tile_nd(
    size: Tuple[int, ...], tile_size: Tuple[int, ...], overlap: Tuple[int, ...]
) -> Iterator[Tuple[Tuple[int, int], ...]]:
    """
    Generate the coordinates for tiling an N-dimensional array, in a minimally
    dynamic fasion. This is done by making the last tile along each dimension
    have range `size[i] - tile_size[i]` Example usage can be found in 
    `ryanirl/ndtiler/examples/*`.

    Args:
        size (Tuple[int, ...]): The size of the array.
        tile_size (Tuple[int, ...]): The size of each tile.
        overlap (Tuple[int, ...]): The overlap between adjacent tiles.

    Yields:
        Iterator[Tuple[Tuple[int, int], ...]]: The coordinates of each tile.
    """
    _check_positive_stride(size, tile_size, overlap)
    _check_tile_size(size, tile_size, overlap)
    _check_size_large(size, tile_size, overlap)
    _check_same_dimensions(size, tile_size, overlap)

    stride = get_stride(tile_size, overlap)
    tile_count = get_tile_count(size, tile_size, overlap)
    for index in _generate_inds(0, [], tile_count):
        tiles = []
        for i, l in enumerate(index):
            # Convert the tile index to the proper shape, then add it to tiles.
            ls = l * stride[i]

            # Dynamic update 
            if ls + tile_size[i] > size[i]:
                ls = size[i] - tile_size[i]

            tiles.append((ls, ls + tile_size[i]))

        yield tuple(tiles)


def get_stride(tile_size: Tuple[int, ...], overlap: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Calculate the step sizes (stride) for tiling given the tile size and overlap.

    Args:
        tile_size (Tuple[int, ...]): The size of each tile.
        overlap (Tuple[int, ...]): The overlap between adjacent tiles.

    Returns:
        Tuple[int, ...]: The step sizes for each dimension.
    """
    return tuple(ts - ov for ts, ov in zip(tile_size, overlap))


def get_overlap_from_stride(tile_size: Tuple[int, ...], stride: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Convenience function to compute the overlap from tile size and stride. 

    Args:
        tile_size (Tuple[int, ...]): The size of each tile.
        stride (Tuple[int, ...]): The striding of tiles.

    Returns:
        Tuple[int, ...]: The step sizes for each dimension.
    """
    return tuple(ts - s for ts, s in zip(tile_size, stride))


def get_overflow(
    size: Tuple[int, ...], tile_size: Tuple[int, ...], overlap: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Calculate the adjument factor (overflow) needed to modify the *size* to
    allow even tiling with parameters `tiling` and `overlap`.

    Args:
        size (Tuple[int, ...]): The size of the array.
        tile_size (Tuple[int, ...]): The size of each tile.
        overlap (Tuple[int, ...]): The overlap between adjacent tiles.

    """
    _check_tile_size(size, tile_size, overlap)
    _check_positive_stride(size, tile_size, overlap)

    return tuple(
        (ts - s) + (ts - ov) * ceil(max(s - ts, 0) / (ts - ov)) 
        for s, ts, ov in zip(size, tile_size, overlap)
    )


def get_tile_count(
    size: Tuple[int, ...], tile_size: Tuple[int, ...], overlap: Tuple[int, ...]
) -> List[int]:
    """
    Calculate the number of tiles needed in each dimension.

    Args:
        size (Tuple[int, ...]): The size of the array.
        tile_size (Tuple[int, ...]): The size of each tile.
        overlap (Tuple[int, ...]): The overlap between adjacent tiles.

    Returns:
        Tuple[int, ...]: The step sizes for each dimension.
    """
    return [
        ceil(1 + (s - ts) / (ts - ov)) 
        for s, ts, ov in zip(size, tile_size, overlap)
    ]


def _generate_inds(
    dim: int, current: List[int], tile_count: List[int]
) -> Iterator[Tuple[int, ...]]:
    """
    Recursively generate indices for tiling.

    Args:
        dim (int): The current dimension being processed.
        current (List[int]): The current indices being generated.
        tile_count (List[int]): The number of tiles in each dimension.

    Yields:
        Iterator[Tuple[int, ...]]: The indices for the tiles.
    """
    if dim == len(tile_count):
        yield tuple(current)
    else:
        for i in range(tile_count[dim]):
            yield from _generate_inds(dim + 1, current + [i], tile_count)


def _check_positive_stride(
    size: Tuple[int, ...], tile_size: Tuple[int, ...], overlap: Tuple[int, ...]
) -> None:
    if not all((ts > ov) for ts, ov in zip(tile_size, overlap)):
        raise ValueError(
            "The overlap must be less than or equal to the tiling size. In other "
            "words, the stride must be positive and non-zero."
        )


def _check_same_dimensions(
    size: Tuple[int, ...], tile_size: Tuple[int, ...], overlap: Tuple[int, ...]
) -> None:
    if (len(size) != len(tile_size)) or (len(size) != len(overlap)):
        raise ValueError(
            "`size`, `tile_size`, and `overlap` must all have the same length "
            "(number of dimensions)."
        )


def _check_size_large(
    size: Tuple[int, ...], tile_size: Tuple[int, ...], overlap: Tuple[int, ...]
) -> None:
    if not all((s >= ts) for s, ts in zip(size, tile_size)):
        raise ValueError(
            "The size must be greater than or equal to the tile size." 
        )


def _check_tile_size(
    size: Tuple[int, ...], tile_size: Tuple[int, ...], overlap: Tuple[int, ...]
) -> None:
    if not all((ts > 0) for ts in tile_size):
        raise ValueError(
            "The tiling size must be greater than zero."
        )


def _check_even_tiling(
    size: Tuple[int, ...], tile_size: Tuple[int, ...], overlap: Tuple[int, ...]
) -> None:
    if not all((s - ts) % (ts - ov) == 0 for s, ts, ov in zip(size, tile_size, overlap)):
        raise ValueError(
            "The array size cannot be evenly tiled with the given tile size and overlap."
        )


