#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#if ENZYME_JAX_ENABLE_TRITON
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif

namespace mlir::enzyme {
#define GEN_PASS_DEF_DETACHTRITONSOFTMAXSHIFTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;

namespace {
#if ENZYME_JAX_ENABLE_TRITON
template <class Combiner>
bool reducesRow(triton::ReduceOp reduce, Value input) {
  if (!reduce || reduce.getNumOperands() != 1 ||
      reduce.getOperand(0) != input || reduce.getAxis() != 0 ||
      cast<RankedTensorType>(input.getType()).getRank() != 1)
    return false;
  Block &block = reduce.getRegion().front();
  auto combine = dyn_cast<Combiner>(block.front());
  return block.getOperations().size() == 2 && combine &&
         combine.getLhs() == block.getArgument(0) &&
         combine.getRhs() == block.getArgument(1) &&
         block.getTerminator()->getOperand(0) == combine.getResult();
}

void detachShift(arith::DivFOp divide) {
  auto exp = divide.getLhs().getDefiningOp<math::ExpOp>();
  auto denominator = divide.getRhs().getDefiningOp<triton::SplatOp>();
  if (!exp || !denominator || !denominator->hasOneUse())
    return;
  auto sum = denominator.getSrc().getDefiningOp<triton::ReduceOp>();
  if (!reducesRow<arith::AddFOp>(sum, exp.getResult()) || !sum->hasOneUse() ||
      !llvm::hasNItems(exp->getUses(), 2))
    return;
  auto centered = exp.getOperand().getDefiningOp<arith::SubFOp>();
  if (!centered || !centered->hasOneUse())
    return;
  auto shift = centered.getRhs().getDefiningOp<triton::SplatOp>();
  if (!shift ||
      !reducesRow<arith::MaxNumFOp>(
          shift.getSrc().getDefiningOp<triton::ReduceOp>(), centered.getLhs()))
    return;

  // Detach only this use. A separately returned maximum still needs its AD.
  OpBuilder builder(centered);
  Value detached = enzyme::IgnoreDerivativesOp::create(
      builder, centered.getLoc(), shift.getType(), shift.getResult());
  centered.getRhsMutable().assign(detached);
}
#endif

struct DetachTritonSoftmaxShiftPass final
    : enzyme::impl::DetachTritonSoftmaxShiftPassBase<
          DetachTritonSoftmaxShiftPass> {
  void runOnOperation() override {
#if ENZYME_JAX_ENABLE_TRITON
    getOperation().walk(detachShift);
#else
    getOperation().emitError("Triton is disabled in this build");
    signalPassFailure();
#endif
  }
};
} // namespace
