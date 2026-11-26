# When should we hoist the projection?

The scalar cell updates `h = decay * h + dot(w, x)` for T steps with a fixed D-wide feature vector. The benchmark asks Enzyme for the gradient of the final state with respect to w, then compiles the result with Triton. The input is shared across time by construction. This is a mechanism test for a static feature in a recurrence.

With loop-invariant code motion only after AD, the reverse loop scales x and adds it to a D-wide gradient each step. With the same Transform sequence before and after AD, the loop accumulates one scalar and scales x once afterward. The lane-arithmetic work changes from O(TD) to O(T+D). Downstream Triton GPU optimizations run in every configuration.

| Feature width | Steps | Post only, µs | Before and after, µs | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 3.33 | 2.59 | 1.29× |
| 512 | 128 | 5.21 | 2.75 | 1.90× |
| 2048 | 128 | 17.66 | 6.65 | 2.65× |
| 8192 | 4 | 73.84 | 74.88 | 0.99× |

These are median CUDA graph times on a GeForce RTX 5090. Each kernel handles 2048 independent sequences. The harness records 15 samples of 25 launches per case, shuffles the schedule order each round, and checks every output against the geometric-series gradient in fp64. Both forms use fp32 AD arithmetic and can round differently.

Run from the repository root with a Triton-enabled Enzyme-JAX compiler and CUDA PyTorch. The output includes the four derivatives, their GPU entry points, PTX and raw timings.

```bash
python scripts/benchmark_ttir_recurrence.py --opt /path/to/enzymexlamlir-opt \
  --out run --widths 128 512 2048 8192 --steps 4 32 128 \
  --rows 2048 --samples 15 --repeats 25
```
