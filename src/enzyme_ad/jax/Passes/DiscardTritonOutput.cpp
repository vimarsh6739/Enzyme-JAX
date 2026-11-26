#include "src/enzyme_ad/jax/Passes/Passes.h"

#if ENZYME_JAX_ENABLE_TRITON
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif

namespace mlir::enzyme {
#define GEN_PASS_DEF_DISCARDTRITONOUTPUTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;

namespace {
struct DiscardTritonOutputPass final
    : enzyme::impl::DiscardTritonOutputPassBase<DiscardTritonOutputPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
#if ENZYME_JAX_ENABLE_TRITON
    auto fn = module.lookupSymbol<triton::FuncOp>(function);
    if (!fn || fn.isExternal()) {
      module.emitError("expected a defined tt.func named '") << function << "'";
      return signalPassFailure();
    }
    if (argument < 0 || argument >= fn.getNumArguments() ||
        !isa<triton::PointerType>(fn.getArgument(argument).getType())) {
      fn.emitError("expected a pointer argument at index ") << argument.getValue();
      return signalPassFailure();
    }

    SmallVector<Value> pointers = {fn.getArgument(argument)};
    SmallVector<triton::StoreOp> stores;
    while (!pointers.empty()) {
      Value pointer = pointers.pop_back_val();
      for (OpOperand &use : pointer.getUses()) {
        Operation *op = use.getOwner();
        if (use.getOperandNumber() == 0) {
          if (isa<triton::AddPtrOp, triton::SplatOp, triton::BroadcastOp,
                  triton::ExpandDimsOp, triton::ReshapeOp, triton::TransOp>(
                  op)) {
            pointers.push_back(op->getResult(0));
            continue;
          }
          if (auto store = dyn_cast<triton::StoreOp>(op)) {
            stores.push_back(store);
            continue;
          }
        }
        op->emitOpError("cannot discard output argument ")
            << argument.getValue() << " of @" << function
            << ": expected only address construction and ordinary stores";
        return signalPassFailure();
      }
    }
    // Validate the full use chain before changing any of it. In particular,
    // never delete a store merely because an individual program does not
    // read its result: another program can access the same location.
    for (triton::StoreOp store : stores)
      store.erase();
#else
    module.emitError("Triton is disabled in this build");
    signalPassFailure();
#endif
  }
};
} // namespace
