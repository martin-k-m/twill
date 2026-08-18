# Twill 1.6.0 -- the completeness release

1.5 made the ecosystem run. 1.6 makes the language stop having pieces missing
from the middle of it.

Four things were true of twill before this release and are not now: an `I64` was
a float, a systems-mode annotation was a comment, a `match` could silently fail
to cover its cases, and two different mistakes in autodiff answered with a zero
instead of an error.

Nothing in numeric mode changes. A program with no `mode systems` line and no
type annotations behaves exactly as it did, which is what the mode gate is for.

## What the release is

**`I64` is a real 64-bit integer.** It was an `f64` that happened to hold an
integer, so it held 53 bits. `9007199254740993` printed as `9007199254740992`,
`MAX_I64 + 1` did not wrap, `shl(1, 63)` could not be represented, and every
hash mixer and 64-bit generator written in twill was quietly wrong. That is
`docs/needs.md` NEEDS-2, the oldest open entry in the file, and it is the reason
`std/random` had to be moved onto the host's generator in 1.5.1.1.

The specification did not move: `docs/language-guide.md` described wrapping
arithmetic, truncating division, dividend-signed modulo and exact bitwise
operations in 1.5, and the implementation now does them. A property test runs
300 random pairs through every operator against Go's `int64`.

**The systems-mode types are checked.** `I64`, `F64`, `Bool`, `Str`, `Bytes`,
`Unit`, `Arr[T]`, `Dict[K, V]`, `Opt[T]`, `Res[T, E]`, a struct by name, an enum
by name and a declared function type are types the checker knows. A definite
mismatch is reported at a binding, an argument, a return, a struct field at
construction and at every later assignment, and an enum payload:

```
lexer.tw:14: shape error: "kind" is declared I64 but the value is Str
lexer.tw:38: shape error: function "peek" returns Opt[I64] but its signature declares I64
```

**A `match` must cover its cases**, and the missing ones are named. Four related
mistakes are reported the same way: a repeated case, an arm after `_`, a `_`
when everything is already handled, and arms naming cases of two enums. This is
the reason to have an enum instead of an integer tag: adding a case now makes
every `match` that has not been updated say so, at check time.

**`?` is checked** — outside a function, in a function that does not return a
`Res` or an `Opt`, on a value that is neither, and in what it yields. A failing
`?` at the top level of a file used to end the program with status 0 and no
message.

**Two silent zeros in autodiff are errors or answers now.** A gradient taken
inside a gradient is refused wherever it is written, not only as a literal
`grad(grad(f))`; and `tensor(list(...))` over values under differentiation is
differentiable rather than returning a matrix of zeros through `jacobian`.

**The rest of the filesystem**, a monotonic clock, a gradient checker in
`std/gradcheck`, `twill doctor`, `twill test --filter`, `:type` and `:shape` in
the REPL, and a formatter that no longer deletes `unit` declarations.

`CHANGELOG.md` has the complete list.

## The candidates

1.6.0 is 1.6.0-rc2 with nothing added, and the two candidates are worth keeping
apart rather than merging.

rc1 was the release: the language work, the checker, the tooling, the
documentation, verified against the nine ecosystem repositories without any of
them having to change.

rc2 is what happened when those nine repositories were then *moved onto* rc1 --
their integer tags rewritten as enums, their `-1` sentinels as `Opt`, their
error-swallowing reads as `Res`. Code that is trying to use a feature finds
things that code merely passing its tests does not. It found four: a match arm
that could not continue onto the next line, ordering that was not type-checked,
an `I64` that could not be saved, and a cross-module `match` that was never
checked at all. None was reachable from twill's own sources.

Between rc2 and this tag the compiler did not change. What changed is where the
evidence comes from: the nine repositories run their suites against it in CI now,
rather than on one developer's machine.

## Evidence

| | |
| --- | --- |
| Go test suite | green, including the Go-versus-self-hosted differential harness |
| Standard library suites | 15 of 15, including the gradient checker's own 19 checks |
| Ecosystem suites | 60 of 60, across all nine repositories, run by their own CI |
| `twill check` over the ecosystem | 10 unresolved names, all primitives that do not exist yet, down from 31 |
| Formatter corpus | 461 files: parses, idempotent, every comment kept, every statement kept |
| Checker false positives on working code | none measured across ten repositories |

## What breaks

Systems-mode code can newly fail to check. That is the release, and every
diagnostic names what it found and what was declared.

Three run-time behaviours change for a program relying on them:

- an `I64` division or modulo by zero is an error rather than an infinity or a
  NaN;
- `%` on two `I64`s takes the sign of the dividend rather than the divisor;
- a failing `?` at the top level of a file stops with a message rather than
  exiting 0.

One name introduced during this release cycle changed before it shipped:
`rename_path` is `rename`.

## What is deliberately not in it

- **User-defined generics.** `Arr[T]`, `Dict[K, V]`, `Opt[T]` and `Res[T, E]`
  are the generic types; `struct Box[T]` does not parse. NEEDS-4.
- **Richer patterns.** A pattern is a case name, one binding, or `_`. No
  literals, no nesting, no guards. An inner value is matched by a second
  `match`, so each is a convenience rather than a hole, and they wait until the
  cost of not having them is measured rather than assumed.
- **The strict checker policy.** `docs/self-hosting.md` section 1.3 asks for a
  systems mode in which a type still unknown at the end of inference is itself
  an error. This release takes the shape checker's policy instead -- report only
  what is certain -- because the strict one makes every unannotated expression
  in a mostly-annotated file an error, and a checker that is wrong often gets
  turned off. NEEDS-49 stays open with the reasoning recorded.
- **A GPU backend.** `docs/gpu-feasibility.md` measured it and said not yet, for
  reasons that have not changed: the workloads are small, `f64` throws away most
  of the card, and the ceiling is lower than it looks.
- **The tracing compiler, on.** It exists, it is bit-exact against the
  interpreter, and it is slower end to end on every program measured. It stays
  off. `docs/CODEGEN.md` section 11 has the numbers and the five attempts that
  did not close the gap.

## Where 1.7 could go

In the order the evidence supports, not the order that sounds best:

1. **Packed narrow storage** (NEEDS-111). Every float is an `f64`, so a narrow
   dtype is a tag and a rounding rule rather than a layout, and quantisation
   shrinks nothing in memory. This is the single largest constraint on what
   twill can run: 7B parameters in `f64` is 56 GB.
2. **The primitives the ecosystem is still asking for**, which `twill check`
   now enumerates exactly: a process interface, a ranged file read, memory
   counters.
3. **User-defined generics**, whose absence shows up as duplicated containers.
4. **An LSP**, which is where the checker's work becomes visible while typing
   rather than at a command line.
5. **The strict systems-mode policy**, once there is enough evidence about how
   often it would be wrong.
