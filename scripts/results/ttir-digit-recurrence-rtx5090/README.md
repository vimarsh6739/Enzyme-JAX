# Does the projection hoist pay off at eight native steps?

The digit classifier sees a different eight-pixel image row at each of eight steps. Each class has a learned projection of that row and a second learned projection of the 16 row and column means of the entire image. Its scalar state updates as `h = 0.75 * h + dot(row, u) + dot(context, w)`. We fit the two sets of weights and a class bias by ridge least squares on 3,823 training digits. The timed kernel computes the squared one-hot losses and accumulates their gradients into the 10 × 16 context weights across the 1,797 test digits. The held-out writers do not appear in training. [Alpaydin and Kaynak, UCI Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49)

The fitted model classified 85.98% of test images correctly. Refitting with context alone gave 81.86%, and refitting with rows alone gave 52.53%. The gradient checks compare both generated kernels against PyTorch autograd through the explicit eight-step recurrence in fp64. Relative L2 error was below 9.5 × 10⁻⁷ for both.

| Transform LICM placement | Reverse loop carry for the shared projection | SM120 FMUL instructions | Registers | Median kernel time |
| --- | --- | ---: | ---: | ---: |
| After AD | 16-element gradient | 36 | 38 | 16.256 µs |
| Before and after AD | Scalar adjoint | 29 | 38 | 16.256 µs |

The instruction counts cover the entire cubin disassembly. The compiler removed seven multiplications. At the native shape the two schedules tied in median kernel time. We randomized order within 101 pairs and timed 16 launches per schedule per pair using one warp per example and class. Each launch consumed independent output seeds and accumulated into an independent gradient buffer.

Run with the `vim/ttir-ad-transform-hooks` Triton branch and a Triton-enabled Enzyme-JAX build. The recorded run used frontend 03bead828c and the seed synchronization fix from 01e78a9cc. The dataset and compiler fingerprints are in `result.json`. The output directory contains both derivative kernels and cubins.

```sh
curl -fLsS https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra -o /tmp/optdigits.tra
curl -fLsS https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes -o /tmp/optdigits.tes
python scripts/benchmark_digit_recurrence.py /tmp/optdigits.tra /tmp/optdigits.tes --out digits
cuobjdump --dump-sass digits/post.cubin | rg -c ' FMUL '
cuobjdump --dump-sass digits/both.cubin | rg -c ' FMUL '
```
