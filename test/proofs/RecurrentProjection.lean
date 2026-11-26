/-
The reverse TTIR can accumulate a vector in the loop, or a scalar that is
multiplied by the input vector after the loop. Check one arbitrary lane x.
Both loops see the same sequence of adjoints s; in the cell, s₀ is the output
seed and sₜ₊₁ = sₜ * decay. The proof works for any such sequence.
-/
namespace RecurrentProjection

def accumulate {α : Type} (zero : α) (add : α → α → α)
    (term : Nat → α) : Nat → α
  | 0 => zero
  | t + 1 => add (accumulate zero add term t) (term t)

theorem factorLane {α : Type} (zero : α) (add mul : α → α → α)
    (zero_mul : ∀ x, mul zero x = zero)
    (add_mul : ∀ a b x, mul (add a b) x = add (mul a x) (mul b x))
    (s : Nat → α) (x : α) (steps : Nat) :
    accumulate zero add (fun t => mul (s t) x) steps =
      mul (accumulate zero add s steps) x := by
  induction steps with
  | zero => exact (zero_mul x).symm
  | succ t ih => simp only [accumulate, ih, add_mul]

-- Count scalar lane adds and multiplies. Constants, splats, and control
-- flow cost zero under this particular model. Hardware timing is separate.
def postOnlyCost (width steps : Nat) := (2 * width + 1) * steps
def preAndPostCost (width steps : Nat) := 2 * steps + width

-- Width and trip count are positive. Writing them as d+1 and t+1 gives the
-- exact savings and includes the equality case width = steps = 1.
theorem savedOps (d t : Nat) :
    postOnlyCost (d + 1) (t + 1) =
      preAndPostCost (d + 1) (t + 1) + ((2 * d + 1) * t + d) := by
  simp only [postOnlyCost, preAndPostCost, Nat.mul_add, Nat.add_mul,
    Nat.mul_one, Nat.one_mul]
  omega

-- A zero-trip loop skips every vector operation in the post-only schedule.
theorem zeroTrips (d : Nat) :
    postOnlyCost d 0 = 0 ∧ preAndPostCost d 0 = d := by
  simp [postOnlyCost, preAndPostCost]

end RecurrentProjection
