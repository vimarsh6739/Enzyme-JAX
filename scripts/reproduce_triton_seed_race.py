"""Count incorrect load-and-clear results on layouts with replicated warps.

Run with CUDA PyTorch and Triton. This isolates the GPU code generated for the
seed read in a store adjoint. No Enzyme frontend is required.
"""

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def center_and_clear(seed, output, width: tl.constexpr, tile: tl.constexpr, synchronize: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, tile)
    offset = row * width + column
    valid = column < width
    value = tl.load(seed + offset, valid, other=0.)
    if synchronize:
        tl.debug_barrier()
    tl.store(seed + offset, 0., valid)
    centered = tl.where(valid, value - tl.sum(value, 0) / width, 0.)
    sum_squares = tl.sum(centered * centered, 0)
    tl.store(output + offset, centered * sum_squares, valid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", default=300, type=int)
    args = parser.parse_args()
    torch.manual_seed(475)
    rows, width = 2048, 127
    source = torch.randn((rows, width), device="cuda")
    centered = source.double() - source.double().mean(1, keepdim=True)
    expected = centered * centered.square().sum(1, keepdim=True)
    output = torch.empty_like(source)
    print("warps,barrier,launches,failed,max_abs_error")
    for warps in (4, 8):
        for synchronize in (False, True):
            failures = 0
            maximum_error = 0.
            for _ in range(args.repeats):
                seed = source.clone()
                center_and_clear[(rows,)](seed, output, width, 128, synchronize, num_warps=warps)
                error = (output.double() - expected).abs().max().item()
                maximum_error = max(maximum_error, error)
                failures += error > 2e-4
            print(f"{warps},{synchronize},{args.repeats},{failures},{maximum_error}")


if __name__ == "__main__":
    main()
