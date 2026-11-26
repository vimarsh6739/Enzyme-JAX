"""Run with Liger at cc29eda8, the Triton post_ad hook and this Enzyme-JAX pass."""

import torch
from liger_kernel.ops import swiglu
from triton.experimental import Const, Duplicated, autodiff


def main():
    a = torch.linspace(-2, 1, 259, device="cuda")
    b = torch.linspace(1, 3, 259, device="cuda")
    u = torch.linspace(-1, 1, 259, device="cuda")
    primal = swiglu._swiglu_forward_kernel
    pipeline = f"discard-triton-output{{function=revdiffe{primal.fn.__name__} argument=4}},canonicalize"
    results = []
    for post in (None, pipeline):
        out = torch.full_like(a, 123.0)
        seed = u.clone()
        da, db = torch.ones_like(a), torch.full_like(b, 3.0)
        compiled = autodiff(primal, post_ad=post)[(1,)](
            Duplicated(a, da), Duplicated(b, db), Duplicated(out, seed),
            Const(259), Const(1.0), 259, BLOCK_SIZE=512, num_warps=4,
        )
        expected = torch.nn.functional.silu(a) * b if post is None else torch.full_like(a, 123.0)
        torch.testing.assert_close(out, expected)
        assert torch.count_nonzero(seed) == 0
        results.append((da, db))
        ir = compiled.asm["ttir"]
        print(post or "baseline", ir.count("tt.store"), "stores,", ir.count("tt.atomic_rmw"), "atomic adds")
    for combined, discarded in zip(*results):
        torch.testing.assert_close(combined, discarded, atol=0, rtol=0)


if __name__ == "__main__":
    main()
