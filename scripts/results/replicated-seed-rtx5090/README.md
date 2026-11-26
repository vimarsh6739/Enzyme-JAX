# Why the store pullback needs a barrier

Reverse AD consumes the output cotangent by loading its old value and clearing its buffer. A 128-element tile on an eight-warp CTA can replicate each element in two warps. For the failing specialization on NVIDIA, PTX reads the seed from both copies and clears from one. A warp can clear a seed before its peer reads it. A later reduction writes both warps' partial results to the same shared scratch slots, exposing the difference.

The fix places a CTA barrier between the seed read and clear for live ordinary store pullbacks. The barrier is outside the element mask, so masked-off lanes also reach it. A constant stored value only clears the seed. The descriptor path uses a different lowering. This change handles the intra-CTA hazard for regular TTIR stores and is conservative for layouts in which each element has one reader.

Reproduce the load-clear-reduce case without Enzyme:

```sh
python scripts/reproduce_triton_seed_race.py --repeats 300
```

On a GeForce RTX 5090 we saw 56/300 incorrect launches with 8 warps, 0/300 with the barrier, and 0/300 at 4 warps. Separately, a generated reverse softmax failed its fp64 reference 88/100 times on 2048 rows of width 127 with 8 warps. The fixed compiler passed all 100 launches. Each launch used fresh output seeds and zeroed input adjoints. A sparse CP gradient also passed for squared and Huber loss with 513 observations, rank 24, four warps and a batch tile of 16. The CSV files record launch counts or per-case errors. The study branches carry the softmax and CP drivers.

Runtime tests used CUDA driver 575.57.08, PyTorch 2.12.0a0+git0d62256, Triton at 5b86f2c73b and Enzyme-JAX base a56c4856e. The fixed softmax TTIR compiled to cubin for sm80 and sm120 and to hsaco for gfx942. The GPU executions here used sm120.
