# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def chunk_right(
    start: int,
    end: int,
    size: int,
    slide: int,
) -> list[torch.Tensor]:
    """
    Generate overlapping chunks of indices moving from left to right.

    Example: start=0, end=10, size=4, slide=2
        chunks: [[0,1,2,3], [2,3,4,5], [4,5,6,7], [6,7,8,9]]
                 ──────►   ──────►   ──────►   ──────►

    Args:
        start: Start index (inclusive)
        end: End index (exclusive)
        size: Number of elements per chunk
        slide: Step size between chunk starts (size - slide = overlap)
    """
    assert 0 < slide <= size, f"Need slide <= size, got {slide} > {size}"

    chunks = []
    chunk_end = start  # Right edge of current chunk

    while chunk_end < end:
        # Advance chunk_end: first chunk tries to be full-sized, subsequent chunks slide
        if not chunks:
            chunk_end = min(start + size, end)  # First chunk: up to 'size' elements
        else:
            chunk_end = min(chunk_end + slide, end)  # Subsequent: slide forward

        # Compute chunk_start: go back 'size' elements, clamped to 'start'
        chunk_start = max(start, chunk_end - size)

        chunks.append(torch.arange(chunk_start, chunk_end))

    return chunks


def chunk_left(
    start: int,
    end: int,
    size: int,
    slide: int,
) -> list[torch.Tensor]:
    """
    Generate overlapping chunks of indices moving from right to left.

    Example: start=0, end=10, size=4, slide=2
        chunks: [[9,8,7,6], [7,6,5,4], [5,4,3,2], [3,2,1,0]]
                 ◄──────   ◄──────   ◄──────   ◄──────

    This is equivalent to chunk_right but with:
        - Chunks returned in reverse order (rightmost first)
        - Each chunk's elements reversed (descending indices)

    Args:
        start: Start index (inclusive)
        end: End index (exclusive)
        size: Number of elements per chunk
        slide: Step size between chunk starts (size - slide = overlap)
    """
    # Generate right-to-left by reversing chunk_right results
    right_chunks = chunk_right(start, end, size, slide)
    return [chunk.flip(0) for chunk in reversed(right_chunks)]


def chunk_from(
    start: int,
    total: int,
    size: int,
    slide: int,
) -> list[torch.Tensor]:
    """
    Generate chunked timesteps starting from a given position, expanding in both directions.

    Args:
        start: Starting timestep index
        total: Total number of timesteps
        size: Number of timesteps per chunk
        slide: Sliding window step size
    """
    context = size - slide

    # Special case: stitch all timesteps into one chunk with start first
    if total == size:
        indices = torch.arange(total)
        return [torch.cat([indices[start : start + 1], indices[indices != start]])]

    # Edge cases: start at boundary
    if start == 0:
        return chunk_right(0, total, size, slide)
    if start == total - 1:
        return chunk_left(0, total, size, slide)

    # Determine primary direction (start with the side that has more timesteps)
    left_first = start > total - start

    if left_first:
        left = chunk_left(0, start + 1, size, slide)
        right_start = min(max(0, start - context + 1), total - size)
        right = chunk_right(right_start, total, size, slide)
        return left + right
    else:
        right = chunk_right(start, total, size, slide)
        left_end = max(min(start + context, total), size)
        left = chunk_left(0, left_end, size, slide)
        return right + left
