"""Time Enzyme's reverse TTIR with loop-invariant code motion before and after AD.

Pass --opt /path/to/enzymexlamlir-opt. The output directory contains the TTIR
and PTX for each schedule, timings.csv (CUDA graph replay), and metadata.json.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess

import torch
import triton


FIXTURE = (Path(__file__).resolve().parents[1] / "test/lit_tests/diffrules/triton"
           / "recurrent_projection_schedule.mlir")
AD = ("--enzyme-wrap=infn=cell outfn=pullback "
      "argTys=enzyme_active,enzyme_const,enzyme_const,enzyme_const "
      "retTys=enzyme_active mode=ReverseModeCombined")
CLEAN = ["--canonicalize", "--remove-unnecessary-enzyme-ops", "--canonicalize", "--cse"]
SCHEDULES = {"none": (False, False), "post": (False, True), "pre": (True, False), "both": (True, True)}


def launch_wrapper(width):
    # One program per sequence. w does not affect this linear VJP.
    return f"""
  tt.func public @kernel(%x: !tt.ptr<f32>, %dweights: !tt.ptr<f32>, %decay: f32, %steps: i32, %seed: f32) {{
    %pid = tt.get_program_id x : i32
    %width = arith.constant {width} : i32
    %row = arith.muli %pid, %width : i32
    %start = tt.splat %row : i32 -> tensor<{width}xi32>
    %idx = tt.make_range {{start = 0 : i32, end = {width} : i32}} : tensor<{width}xi32>
    %offset = arith.addi %start, %idx : tensor<{width}xi32>
    %xbase = tt.splat %x : !tt.ptr<f32> -> tensor<{width}x!tt.ptr<f32>>
    %xptr = tt.addptr %xbase, %offset : tensor<{width}x!tt.ptr<f32>>, tensor<{width}xi32>
    %features = tt.load %xptr : tensor<{width}x!tt.ptr<f32>>
    %w = arith.constant dense<0.0> : tensor<{width}xf32>
    %grad = tt.call @pullback(%w, %features, %decay, %steps, %seed) : (tensor<{width}xf32>, tensor<{width}xf32>, f32, i32, f32) -> tensor<{width}xf32>
    %out = tt.splat %dweights : !tt.ptr<f32> -> tensor<{width}x!tt.ptr<f32>>
    %dest = tt.addptr %out, %offset : tensor<{width}x!tt.ptr<f32>>, tensor<{width}xi32>
    tt.store %dest, %grad : tensor<{width}x!tt.ptr<f32>>
    tt.return
  }}
"""


def build(opt, out, width, schedule):
    def optimize(source, *flags):
        return subprocess.run([opt, *flags], input=source, text=True, check=True, stdout=subprocess.PIPE).stdout

    before, after = SCHEDULES[schedule]
    source = FIXTURE.read_text().replace("tensor<128x", f"tensor<{width}x")
    pre = ["--transform-interpreter"] if before else []
    post = ["--transform-interpreter"] if after else []
    derivative = optimize(source, *pre, AD, *CLEAN, *post, "--enzyme-hlo-remove-transform")
    (out / "pullback.mlir").write_text(derivative)
    # Link the generated private function with a public GPU entry point.
    module = derivative.rstrip().removesuffix("}") + launch_wrapper(width) + "}"
    ttir = out / "kernel.ttir"
    ttir.write_text(optimize(module, "--inline", "--symbol-dce", "--canonicalize", "--cse"))
    kernel = triton.compile(str(ttir), options={"num_warps": 4})
    (out / "kernel.ptx").write_text(kernel.asm["ptx"])
    return kernel


def time_graph(graph, repeats):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", default="enzymexlamlir-opt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--widths", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--steps", nargs="+", type=int, default=[4, 32, 128])
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    if (args.rows <= 0 or args.samples <= 0 or args.repeats <= 0 or min(args.steps) < 0
            or any(w <= 0 or w & (w - 1) for w in args.widths)):
        parser.error("use positive counts, nonnegative steps and power-of-two widths")

    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(23)
    torch.manual_seed(23)
    decay = 0.984375
    seed = -0.625
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": str(torch.__version__),
        "triton": triton.__version__,
        "opt": args.opt,
        "opt_version": subprocess.check_output([args.opt, "--version"], text=True).strip(),
        "study_commit": subprocess.check_output(["git", "-C", str(FIXTURE.parent), "rev-parse", "HEAD"], text=True).strip(),
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "seed": seed, "decay": decay, "random_seed": 23, "warps": 4, "replays_per_sample": args.repeats,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.out / "timings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["schedule", "rows", "width", "steps", "sample", "us", "max_abs_error", "registers"])
        for width in args.widths:
            x = torch.randn((args.rows, width), device="cuda")
            y = torch.empty_like(x)
            kernels = {}
            for name in SCHEDULES:
                path = args.out / f"D{width}-{name}"
                path.mkdir()
                kernels[name] = build(args.opt, path, width, name)
            for steps in args.steps:
                # Independent closed form for dh[T]/dw in double precision.
                expected = x.double() * (seed * (1 - decay**steps) / (1 - decay))
                graphs = {}
                errors = {}
                for name, kernel in kernels.items():
                    launch = kernel[(args.rows, 1, 1)]
                    # Poison the output to detect missing stores.
                    y.fill_(float("nan"))
                    launch(x, y, decay, steps, seed)
                    torch.testing.assert_close(y.double(), expected, rtol=1e-5, atol=2e-5)
                    errors[name] = (y.double() - expected).abs().max().item()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.repeats):
                            launch(x, y, decay, steps, seed)
                    graph.replay()
                    graphs[name] = graph
                torch.cuda.synchronize()
                names = list(kernels)
                for sample in range(args.samples):
                    rng.shuffle(names)
                    for name in names:
                        us = time_graph(graphs[name], args.repeats)
                        writer.writerow([name, args.rows, width, steps, sample, us, errors[name], kernels[name].n_regs])
                handle.flush()


if __name__ == "__main__":
    main()
