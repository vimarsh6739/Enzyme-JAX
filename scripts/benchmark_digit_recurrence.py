"""Fit a row-sequence optical digit classifier and time its generated weight gradient."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import random
import statistics

import numpy as np
import torch
import triton
import triton.language as tl
from triton.experimental import Const, Duplicated, TTIRTransform, autodiff


DECAY = 0.75
HOIST = TTIRTransform("""module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %loops : !transform.any_op
    transform.yield
  }
}""")


@triton.jit
def digit_loss(W, C, Rows, U, Bias, Labels, Loss):
    # Each program contributes one example to one class's weight gradient.
    image = tl.program_id(0)
    cls = tl.program_id(1)
    context_lane = tl.arange(0, 16)
    pixel = tl.arange(0, 8)
    w = tl.load(W + cls * 16 + context_lane)
    context = tl.load(C + image * 16 + context_lane)
    u = tl.load(U + cls * 8 + pixel)
    h = 0.
    for t in range(8):
        row = tl.load(Rows + image * 64 + t * 8 + pixel)
        h = 0.75 * h + tl.sum(row * u, 0) + tl.sum(context * w, 0)
    score = h + tl.load(Bias + cls)
    residual = score - (tl.load(Labels + image) == cls).to(tl.float32)
    tl.store(Loss + image * 10 + cls, 0.5 * residual * residual)


def load_split(path, size):
    data = np.loadtxt(path, delimiter=",")
    if data.shape != (size, 65):
        raise ValueError(f"expected {size} rows and 65 columns in {path}")
    image = torch.from_numpy(data[:, :64].copy()).reshape(-1, 8, 8) / 16.
    labels = torch.from_numpy(data[:, -1].astype(np.int64))
    return image, labels


def context(image):
    return torch.cat((image.mean(1), image.mean(2)), dim=1)


def features(image):
    # Fit the linear recurrence exactly by solving ridge least squares.
    # The test reference below executes the recurrence directly.
    powers = torch.tensor([DECAY**i for i in reversed(range(8))], dtype=torch.float64)
    return torch.cat((powers.sum() * context(image), (image * powers[None, :, None]).sum(1),
                      torch.ones((len(image), 1), dtype=torch.float64)), dim=1)


def fit(design, labels):
    target = torch.nn.functional.one_hot(labels, 10).double()
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * 0.01
    penalty[-1, -1] = 0.
    return torch.linalg.solve(design.T @ design + penalty, design.T @ target)


def fp64_reference(image, labels, w, u, bias):
    weights = w.detach().requires_grad_(True)
    ctx = context(image)
    h = torch.zeros((len(image), 10), dtype=torch.float64)
    for t in range(8):
        h = DECAY * h + image[:, t] @ u.T + ctx @ weights.T
    score = h + bias
    loss = 0.5 * (score - torch.nn.functional.one_hot(labels, 10)).square()
    grad, = torch.autograd.grad(loss.sum(), weights)
    return score.detach(), loss.detach(), grad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path, help="UCI optdigits.tra")
    parser.add_argument("test", type=Path, help="UCI optdigits.tes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--slots", type=int, default=16)
    args = parser.parse_args()
    if min(args.samples, args.slots) < 1:
        parser.error("--samples and --slots must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    train, train_y = load_split(args.train, 3823)
    images, labels = load_split(args.test, 1797)
    train_design = features(train)
    params = fit(train_design, train_y)
    # Use the parameters after casting to f32 for the independent oracle.
    w, u, bias = (p.float().contiguous().cuda() for p in (params[:16].T, params[16:24].T, params[-1]))
    scores, losses, dw_ref = fp64_reference(images, labels, w.cpu().double(), u.cpu().double(), bias.cpu().double())
    # Refit a context-only model to check how much the ordered rows contribute.
    ctx_train = torch.cat((train_design[:, :16], train_design[:, -1:]), 1)
    test_design = features(images)
    ctx_test = torch.cat((test_design[:, :16], test_design[:, -1:]), 1)
    ctx_scores = ctx_test @ fit(ctx_train, train_y)
    row_scores = test_design[:, 16:] @ fit(train_design[:, 16:], train_y)
    metadata = {
        "train_images": len(train), "test_images": len(images), "steps": 8, "pixels_per_row": 8,
        "context_width": 16, "classes": 10, "ridge": 0.01, "decay": DECAY,
        "test_accuracy": (scores.argmax(1) == labels).double().mean().item(),
        "context_only_test_accuracy": (ctx_scores.argmax(1) == labels).double().mean().item(),
        "rows_only_test_accuracy": (row_scores.argmax(1) == labels).double().mean().item(),
        "test_mean_loss_per_image": losses.sum(1).mean().item(),
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(args.test.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "triton": triton.__version__,
    }
    from enzyme_ad.jax import enzyme_call

    with Path(enzyme_call.__file__).open("rb") as lib:
        metadata["enzyme_binding_sha256"] = hashlib.file_digest(lib, "sha256").hexdigest()
    metadata["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                                 text=True).splitlines()[0]
    metadata.update(warps=1, pairs=args.samples, launches_per_sample=args.slots, shuffle_seed=23)
    ctx = context(images).float().contiguous().cuda()
    rows = images.float().contiguous().cuda()
    labels = labels.int().cuda()
    output = torch.empty((len(images), 10), device="cuda")
    dw = torch.empty_like(w)
    seed = torch.empty_like(output)

    def inputs(weight_grad, loss_seed):
        return (Duplicated(w, weight_grad), Const(ctx), Const(rows), Const(u), Const(bias),
                Const(labels), Duplicated(output, loss_seed))

    batch_dw = torch.empty((args.slots, *dw.shape), device="cuda")
    batch_seed = torch.empty((args.slots, *seed.shape), device="cuda")
    slots = [inputs(g, s) for g, s in zip(batch_dw.unbind(), batch_seed.unbind())]
    grid = (len(images), 10)
    graphs = {}
    errors = {}
    registers = {}
    for name, pre in [("post", None), ("both", HOIST)]:
        diff = autodiff(digit_loss, pre_ad=pre, post_ad=HOIST)
        dw.zero_()
        seed.fill_(1)
        output.fill_(float("nan"))
        kernel = diff[grid](*inputs(dw, seed), num_warps=1)
        registers[name] = kernel.n_regs
        torch.testing.assert_close(output.cpu().double(), losses, atol=4e-5, rtol=1e-4)
        rel = (dw.cpu().double() - dw_ref).norm().item() / dw_ref.norm().item()
        if rel >= 1e-4 or torch.count_nonzero(seed).item():
            raise ValueError(f"{name} VJP or seed reset failed (relative gradient error {rel})")
        errors[name] = rel
        (args.out / f"{name}.ttir").write_text(kernel.asm["ttir"])
        (args.out / f"{name}.cubin").write_bytes(kernel.asm["cubin"])
        # CUDA graph timestamp nodes exclude buffer initialization and Python dispatch.
        start = torch.cuda.Event(enable_timing=True, external=True)
        stop = torch.cuda.Event(enable_timing=True, external=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            batch_dw.zero_()
            batch_seed.fill_(1)
            start.record()
            for tensors in slots:
                diff[grid](*tensors, num_warps=1)
            stop.record()
        for _ in range(10):
            graph.replay()
        graphs[name] = graph, start, stop

    torch.cuda.synchronize()
    rng = random.Random(23)
    times = {name: [] for name in graphs}
    with (args.out / "samples.csv").open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pair", "position", "schedule", "us"])
        for pair in range(args.samples):
            names = list(graphs)
            rng.shuffle(names)
            for position, name in enumerate(names):
                graph, start, stop = graphs[name]
                graph.replay()
                torch.cuda.synchronize()
                us = start.elapsed_time(stop) * 1000 / args.slots
                times[name].append(us)
                writer.writerow([pair, position, name, us])
    metadata["relative_l2_gradient_error"] = errors
    metadata["registers"] = registers
    metadata["kernel_us_median"] = {name: statistics.median(values) for name, values in times.items()}
    metadata["median_paired_post_over_both"] = statistics.median(a / b for a, b in zip(times["post"], times["both"]))
    # Resample entire pairs so both schedules keep their measurement order.
    ratios = np.array(times["post"]) / np.array(times["both"])
    samples = np.random.default_rng(23).choice(ratios, size=(10000, len(ratios)), replace=True)
    metadata["paired_ratio_bootstrap_95pct"] = np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()
    (args.out / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
