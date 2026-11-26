"""Try the same VJP before and after Triton's NVIDIA lowering. No GPU is required."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.autodiff import _find_tt_func, _make_public


@triton.jit
def projection(w, x, y, decay, steps, width: tl.constexpr):
    lane = tl.arange(0, width)
    weights = tl.load(w + lane)
    features = tl.load(x + lane)
    state = 0.
    for _ in range(steps):
        state = decay * state + tl.sum(weights * features, 0)
    tl.store(y, state)


# Apply the same activities as the TTIR path. The NVIDIA entry ABI has two
# extra scratch/profile pointers. Neither buffer is used by this primal.
LLVM_VJP = """
declare void @__enzyme_autodiff(ptr, ...)

define void @vjp(ptr addrspace(1) %w, ptr addrspace(1) %dw, ptr addrspace(1) %x,
                 ptr addrspace(1) %y, ptr addrspace(1) %dy, float %decay, i32 %steps) {
  call void (ptr, ...) @__enzyme_autodiff(ptr @projection,
      metadata !"enzyme_dup", ptr addrspace(1) %w, ptr addrspace(1) %dw,
      metadata !"enzyme_const", ptr addrspace(1) %x,
      metadata !"enzyme_dup", ptr addrspace(1) %y, ptr addrspace(1) %dy,
      metadata !"enzyme_const", float %decay, metadata !"enzyme_const", i32 %steps,
      metadata !"enzyme_const", ptr addrspace(1) null, metadata !"enzyme_const", ptr addrspace(1) null)
  ret void
}
"""

SCALAR_CONTROL = """
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @projection(ptr addrspace(1) %w, ptr addrspace(1) %x, ptr addrspace(1) %y,
                                  float %decay, i32 %steps, ptr addrspace(1) %scratch, ptr addrspace(1) %profile) {
  %weight = load float, ptr addrspace(1) %w
  %feature = load float, ptr addrspace(1) %x
  %dot = fmul float %weight, %feature
  store float %dot, ptr addrspace(1) %y
  ret void
}
"""


def try_llvm_ad(args, name, source):
    ir = args.out / f"{name}.ll"
    ir.write_text(source + LLVM_VJP)
    subprocess.run([args.llvm_opt, "-passes=verify", "-disable-output", ir], check=True)
    run = subprocess.run([args.llvm_opt, "-load-pass-plugin", args.plugin, "-passes=enzyme,verify",
                          "-S", ir, "-o", args.out / f"{name}-vjp.ll"], capture_output=True, text=True)
    (args.out / f"{name}.stderr").write_text(run.stderr)
    intrinsics = re.findall(r"^cannot handle \(reverse\) unknown intrinsic\n([^\n]+)", run.stderr, re.MULTILINE)
    assertion = next((line.split("Assertion ", 1)[1] for line in run.stderr.splitlines() if "Assertion " in line), None)
    return {"returncode": run.returncode, "unsupported_intrinsics": dict(Counter(intrinsics)), "assertion": assertion}


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ttir-opt", type=Path, required=True)
    parser.add_argument("--llvm-opt", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True, help="LLVMEnzyme shared library for this opt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    target = GPUTarget("cuda", 120, 32)
    options = {"num_warps": 4}
    source = ASTSource(projection, signature=dict(w="*fp32", x="*fp32", y="*fp32", decay="fp32", steps="i32"),
                       constexprs={"width": 128})
    raw = triton.compile(source, target=target, options=options).asm["ttir"]
    ttir = subprocess.check_output([args.ttir_opt, "--strip-debuginfo", "--loop-invariant-code-motion"],
                                   input=raw, text=True)
    primal = args.out / "projection.ttir"
    primal.write_text(ttir)

    wrap = ("--enzyme-wrap=infn=projection outfn=gradient "
            "argTys=enzyme_dup,enzyme_const,enzyme_dup,enzyme_const,enzyme_const mode=ReverseModeCombined")
    derivative = subprocess.check_output([args.ttir_opt, wrap, "--canonicalize", "--remove-unnecessary-enzyme-ops",
                                          "--canonicalize", "--cse"], input=ttir, text=True)
    pullback = args.out / "gradient.ttir"
    pullback.write_text("module {\n" + _make_public(_find_tt_func(derivative, "gradient")) + "\n}\n")
    gpu = triton.compile(str(pullback), target=target, options=options)
    (args.out / "gradient.cubin").write_bytes(gpu.asm["cubin"])

    llvm = triton.compile(str(primal), target=target, options=options).asm["llir"]
    positive = try_llvm_ad(args, "scalar-control", SCALAR_CONTROL)
    if positive["returncode"]:
        raise RuntimeError(f"LLVM Enzyme failed the scalar ABI control; see {args.out / 'scalar-control.stderr'}")
    result = {
        "target": "sm120", "warps": 4, "triton": triton.__version__, "enzyme_plugin_sha256": sha256(args.plugin),
        "ttir_opt_sha256": sha256(args.ttir_opt), "llvm_opt_sha256": sha256(args.llvm_opt),
        "llvm_version": subprocess.check_output([args.llvm_opt, "--version"], text=True).strip(),
        "input_ttir_sha256": sha256(primal), "ttir_ad_cubin_bytes": len(gpu.asm["cubin"]),
        "scalar_llvm_ad": positive, "lowered_llvm_ad": try_llvm_ad(args, "lowered", llvm),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
