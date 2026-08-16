# Exercises

Questions I should be able to answer cold about twill, without opening the source.
Ordered easy to hard. Answers are in
[EXERCISES-ANSWERS.md](EXERCISES-ANSWERS.md), in a separate file so this one can
be handed to someone else.

Every question is grounded in code that exists and in numbers from
[BENCHMARKS.md](BENCHMARKS.md), [BUGS.md](BUGS.md), [CORRECTNESS.md](CORRECTNESS.md)
or [DECISIONS.md](DECISIONS.md).

---

## Part 1: ten questions

**1.** `grad` is reverse-mode and `hessian` runs forward-mode jets. Give the cost
argument for each choice in terms of input and output count, and say why both modes
exist in one system.

**2.** twill has no tape object. Say what the reverse graph actually is, what makes
the wiring free when no gradient is wanted, and name the three concrete costs of
having no tape.

**3.** In the profile of the differentiated Monte Carlo pricer, the interpreter is
0.035% of flat time. Say what the other numbers are, and state which one is the
actionable finding and what design it argues for.

**4.** twill reaches 21.5 GFLOP/s f64 at 1024x1024 and PyTorch reaches 250. Name the
four things that account for the overall gap, and say which of the four that
particular ratio is.

**5.** The gradient check has 103 cases, but the number that matters is a different
one. Say what `TestGradientCheckCoversEveryOperator` does, what the three counts
are, and what property it buys.

**6.** The finite-difference cotangent is `sin(1.7i + 0.3) * (1 + 0.25(i mod 5))`
rather than all ones. Explain what an all-ones cotangent would hide, and name the
operator families it would hide it for.

**7.** `grad` through `linear(x, quantize(W))` returned exactly zero while returning
the right value. Give the mechanism, and explain why the reasoning that produced the
defect was sound while its conclusion was not.

**8.** The shape checker makes one claim and declines another. State both precisely,
give the six-line program that separates them, and say why the corpus that scores it
100% cannot establish the claim it declines.

**9. (design)** Section 5 of `BENCHMARKS.md` measures 18% of runtime in allocating
and zeroing intermediates, and `DECISIONS.md` entry 4 says the change is not a VM
but an IR. Design the smallest IR that lets the eight elementwise operations of
`mc_option_grad` fuse into one pass. Say what a node is given that `value.Value` is
`any`, where the boundary between compiled and interpreted code sits, how the
backward pass of a fused chain is represented when the tape *is* the tensor graph,
what a fused kernel is cached against, and what determinism rule the fusion must not
break.

**10. (design)** Entry 5 says every float is an f64 and calls it the largest cost in
the document. Design the move to packed storage. Say what a tensor becomes, what
happens to the single-loop kernels, how the accumulation rules in `docs/dtypes.md`
survive the change, what it does to the self-hosted differential harness that
compares canonical float renderings byte for byte, and in what order you would do it
so the numerics stay checkable at every step. Then say what you would measure to
know it worked, and what you expect it *not* to fix.

---

## Part 2: predict the failure

For each scenario, say what twill does, and why. "Why" means the mechanism.

**Scenario A: a program computes `y = round(x) * x`, differentiates it with `grad`,
and then asks for `hessian` of the same function.**

What does `grad` return, and is it a number or an error? What did `hessian` used to
do here, what does it do now, and what is the correct second derivative of a
function that does not depend on its input? Name the property that `floor`, `ceil`,
`round` and the comparisons share, and say why it makes them the same class of
hazard as the quantised kernels were.

**Scenario B: a program reads a matrix from a CSV, multiplies it by a vector whose
length is a literal, and `twill check` reports no problems.**

Does the program run? What decides that? Name the type the checker assigns and the
rule that makes it stay silent, and say whether staying silent is a bug. Then say
what would have to change about the language, not the checker, for this case to be
decidable.

**Scenario C: the self-hosted implementation sums a 10,000-element tensor and the
golden test fails on the last digits of a printed float.**

What differs between the two implementations? Why is that a test failure rather
than a curiosity? Name the decision that makes one of the two answers wrong rather
than merely different, and say what the fix had to port. Then say what would happen
to the self-hosting programme if that decision had gone the other way.

---

## Part 3: delete it and write it again

**Component: `internal/tensor/scan.go`.**

Delete it and reimplement it from scratch. Keep the tests. It is four operations
with four backward passes, its correctness is checked against finite differences by
a harness that will not let it be skipped, and two of its four backwards have a
non-obvious right answer.

You are reimplementing, as **tracked** operations:

- `cumsum`, `cumprod`, `cummax`, `cummin`, forward and backward.
- `cumprod`'s backward as the **division-free prefix/suffix pair**, so a zero in the
  series gives an exact answer rather than `0/0`.
- `cummax` and `cummin` recording the argmax forward and scattering to it, with
  **ties going to the earlier element**, matching `max` and `argmax`.

**Verification.** A correct reimplementation passes:

```sh
go test ./internal/tensor/ -run TestGradient -count=1
go test ./internal/interp/ -run TestCumulative -count=1
```

The first is the one that matters, and it contains two separate traps. The full
operator gradient check will fail with "no gradient reached the leaf" if you return
a bare `&Tensor` from any of the four, which is the exact shape of both bug 1 and
bug 3 in `BUGS.md`. And `TestGradientCheckCoversEveryOperator` parses the package
source, so a fifth scan added without a case or a written exemption fails the suite
whether or not anyone remembers to add one.

Then confirm the composite path, because a scan in isolation is the easy case:

```sh
go test ./internal/interp/ -count=1
```

`TestCumulativeGradientThroughDrawdown` is the one to watch. `max_drawdown` divides
by a `cummax` peak, so before the fix its gradient was neither correct nor obviously
broken, and `sma`, `equity`, `total_return` and `cagr` in `std/backtest` were all in
that state.
