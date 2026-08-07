# Twill language guide

This is the reference for Twill v0.28. The language is small, so this is short.

## Running programs

```bash
twill path/to/program.tw    # shape-check, then run
twill run path/to/program.tw
twill check path/to/program.tw   # shape-check only
twill fmt path/to/program.tw     # canonically format (add --write to edit in place)
twill                             # REPL
```

`twill fmt` reprints a program in a canonical style, preserving comments. It
refuses rather than move a comment it can't place.

Pass `--no-check` to run without the static shape check. In the REPL, each line's
value is printed; `:help` and `:quit` do the obvious things.

## Lexical structure

- Comments run from `#` to end of line.
- Whitespace is insignificant, with one exception: a token that could either
  continue the previous line or start a new one begins a new statement when it
  appears at the start of a line. This applies to a leading `+`/`-` (which would
  otherwise read as subtraction) and to a leading `(`/`[` (which would otherwise
  read as a call or index). To continue an expression across lines, end the line
  with the operator, or keep the call/index on the same line as its target.
- Identifiers match `[A-Za-z_][A-Za-z0-9_]*`.
- Numbers are floating point: `3`, `3.14`, `1e-3`, `.5`.
- Strings use double quotes with `\n`, `\t`, `\"`, `\\` escapes.

## Values

| Type | Example | Notes |
| --- | --- | --- |
| Tensor | `3.0`, `[1.0, 2.0]`, `[[1.0],[2.0]]` | The core type. Scalars are rank-0 tensors. |
| Bool | `true`, `false` | From comparisons and logic. |
| String | `"hello"` | For `print` and messages. |
| List | `range(5)`, `[grad(f), 2]` | Heterogeneous; from `[...]` of non-numbers, `list(...)`, or `range`. |
| Record | `{ w: [1.0], b: 0.0 }` | Named fields; access with `.`. |
| Function | `fn(x) = x + 1` | Closures capture their scope. |
| Unit | `()` | The result of `print`, loops, etc. |

A bracketed literal whose elements are all numbers (or nested numeric brackets)
is a tensor. If any element isn't numeric, it's a list. Build a tensor from
computed values with `tensor([...])`.

```rust
[1.0, 2.0, 3.0]           # tensor, shape [3]
[[1.0, 2.0], [3.0, 4.0]]  # tensor, shape [2, 2]
[grad(f), "x", true]      # list
```

## Systems-mode types

A file whose first non-comment line is `mode systems` gets the subset designed
in `docs/self-hosting.md`. Annotation is mandatory there, so every type has to
have a name that can be written. This is the complete list of those names.

| Name | What it is |
| --- | --- |
| `I64` | signed 64-bit integer, two's complement, wrapping |
| `F64` | IEEE 754 binary64 scalar, **not** a rank-0 tensor |
| `Bool` | `true` or `false`, the result of a comparison or of `and`/`or`/`not` |
| `Byte` | an `I64` constrained to 0..255 at construction |
| `Bytes` | a growable, mutable byte sequence |
| `Str` | an immutable byte string, O(1) length and byte indexing |
| `Arr[T]` | a growable, mutable, homogeneous array |
| `Dict[K, V]` | a hash map, `K` is `Str` or `I64`, iterated in insertion order |
| `Opt[T]`, `Res[T, E]` | the two standard enums |
| a `struct` name | a nominal record declared with `struct` |
| `Unit` | the type of `()` |

`docs/self-hosting.md` section 1.2 lists all of these except `Bool` and `F64`,
and section 1.3 makes annotation mandatory, which is a contradiction rather than
an omission: a `Bool` field cannot be declared and a `F64` cannot be declared,
so no file that has either can be written at all. Both are named here.

### `Bool`

`Bool` is a type name in systems mode, spelled exactly like that, and it is
legal anywhere a type is: a parameter, a return type, a `struct` field, a `let`
annotation, and an `Arr` or `Dict` element type.

```rust
struct Tok { kind: I64, text: Str, trailing: Bool }

fn is_space(c: I64) -> Bool = c == 32 or c == 9
```

There is no conversion between `Bool` and `I64` in either direction, and none is
implied by a comparison. `if` and `while` take a `Bool` and nothing else, so an
`I64` used as a condition is a checker error rather than a test against zero.
Numeric mode is unaffected: there a comparison still yields the `Bool` value
described under [Values](#values), and `Bool` is not a name the checker knows,
because numeric mode has no annotations to write it in.

### `F64`, and what a systems-mode scalar is

**A systems-mode scalar is not a tensor.** `F64` is a plain immutable 64-bit
float, held by value, with no shape, no gradient, no tape entry, and no
allocation. `mode systems` has no tensor type at all, so a rank-0 tensor is not
merely discouraged there, it cannot be named.

This is the answer to a question that was priced before it was asked. loom's
`src/metrics.tw` accumulates a running total once per training step. If a
systems-mode scalar were a rank-0 tensor, every `total = total + x` would
allocate a tensor and a tape node, and an epoch would build a chain of them; the
accumulator would cost more than the model. It does not. `Meter.total: F64` is a
machine word and `+` on it is an instruction.

The two halves of the language are separated by a conversion and not by a
coincidence of representation:

- `f64(n)` widens an `I64` to `F64`, losing precision above 2^53.
- `i64(x)` narrows an `F64` to `I64`, **truncating toward zero**, and is
  undefined outside the `I64` range.
- There is no implicit conversion in either direction, ever, and no implicit
  conversion between `F64` and a numeric-mode tensor. Crossing that seam is
  entry 17 of `docs/roadmap.md` and is deliberately not answered here.

`F64` carries the ordinary IEEE rules, which are the same rules numeric mode
already has: `NaN != NaN`, division by zero gives an infinity rather than
failing, and `-0.0` and `0.0` compare equal while being different values. The
comparison operators are defined on `F64` and return `Bool`.

Arithmetic is `+ - * /` and `%`. `%` on `F64` is the floored modulo numeric mode
already uses (see [Integer division and modulo](#integer-division-and-modulo-on-i64)
for why the `I64` rule is different). The transcendental functions on `F64` are
entry 15 of the roadmap and are not specified here; what is specified is that
when they arrive they take and return `F64`, not a rank-0 tensor.

Numeric mode is unaffected. There a scalar is still a rank-0 tensor, which is
principle 1 in `docs/design.md` and is what keeps autodiff, broadcasting and
printing uniform. The two answers differ because the two modes want different
things, and the mode gate is what lets both be true.

## Operators

Lowest to highest precedence:

| Operators | Meaning |
| --- | --- |
| `or` / `\|\|`, `and` / `&&` | short-circuiting logic |
| `==` `!=` `<` `<=` `>` `>=` | comparison (returns Bool); see [Equality](#equality) |
| `+` `-` | add / subtract (elementwise) |
| `*` `/` `%` `@` | multiply / divide / modulo (elementwise), matmul (`@`) |
| `^` | power (right-associative, scalar exponent) |
| unary `-`, `not` / `!` | negation, logical not |

Elementwise operators broadcast NumPy-style: a scalar against a tensor, a row
vector across a matrix, a column vector down its rows, and so on. Two shapes
combine when, aligned from the right, each pair of dimensions is equal or one of
them is 1. `@` covers vector·vector (dot), matrix·vector, vector·matrix, and
matrix·matrix.

### Bitwise operators on `I64`

These belong to `mode systems` and to `I64` only. There is no bitwise operator
on `F64` and no unsigned integer type. They are spelled as calls rather than as
infix operators, because `and`, `or` and `not` are already the short-circuiting
logical operators and giving one spelling two meanings by operand type is worse
than a name.

| Call | Meaning |
| --- | --- |
| `and(a, b)` | bitwise AND |
| `or(a, b)` | bitwise OR |
| `xor(a, b)` | bitwise XOR |
| `not(a)` | bitwise complement, every bit flipped |
| `shl(a, k)` | left shift, zeros shifted in at the bottom |
| `shr(a, k)` | **arithmetic** right shift, the sign bit shifted in at the top |

`I64` is two's complement and exactly 64 bits. `and`, `or`, `xor` and `not` are
defined bit by bit on that representation and have nothing to say about sign.
`shl` discards bits shifted off the top, so it wraps, and it is the same
operation for negative and non-negative operands.

**`shr` is arithmetic, not logical.** With a negative left operand it shifts
copies of the sign bit in from the top, so `shr(-8, 1)` is `-4` and `shr(-1, k)`
is `-1` for every `k`. Equivalently, `shr(a, k)` is `floor(a / 2^k)` for every
`a`, which is what makes it the right shift for arithmetic and the wrong one for
bit manipulation.

This matters more often than it looks, because the subset has no unsigned type,
so any 64-bit quantity that is conceptually unsigned is carried in an `I64` and
has its top bit set half the time. Hash mixers, IEEE 754 bit patterns and
multiprecision limbs are all in that position.

**Shift counts.** `k` is masked to its low six bits, so the effective count is
`k and 63`. A count of 64 or more therefore wraps rather than saturating to zero
or to the sign, and a negative count is masked into the same 0..63 range rather
than shifting the other way. Both are almost always a bug at the call site; the
masking exists so the operation is total and platform-independent, not because
either is a useful thing to write. Do not rely on it. Where a shift count is
computed, range-check it.

#### Getting a logical right shift

There is no `ushr` operator. Build one. `src/float.tw`'s `ushr` is the idiom,
and it is what every caller in the ecosystem should use or copy:

```rust
let SIGN_BIT: I64 = shl(1, 63)

fn ushr(x: I64, k: I64) -> I64 {
  if k == 0 { return x }
  if x >= 0 { return shr(x, k) }
  or(shr(and(x, not(SIGN_BIT)), k), shl(1, 63 - k))
}
```

Clearing the sign bit makes the value non-negative, where `shr` is already the
logical shift, and the bit is then put back at the position it would have
shifted to. The `k == 0` guard is not decoration: without it `shl(1, 63 - k)`
would be `shl(1, 63)`, which sets the sign bit rather than clearing nothing.

The same construction appears in `std/random.tw` for splitmix64 and xoshiro.
Anything porting a reference implementation written over `uint64` needs it,
because such a reference's `>>` is logical and `shr` is not.

Related: `docs/needs.md` NEEDS-2 (the type and its operators) and NEEDS-85 (why
this is specified here rather than left to the eventual implementation).

### Integer division and modulo on `I64`

When both operands are `I64`, `/` is integer division and `%` is its remainder.
Neither promotes to `F64` and neither yields a fraction: `7 / 2` is `3`, not
`3.5`. Mixing an `I64` and an `F64` is a checker error, not an implicit widening,
so `f64()` or `i64()` has to be written at the seam.

**Rounding.** `/` truncates toward zero and `%` takes the sign of the
**dividend**. That is Go's rule, C99's rule and Rust's rule, and the identity it
preserves is the one worth naming:

```
(a / b) * b + a % b == a        for every a, and every b that is not 0
```

| Expression | Value |
| --- | --- |
| `7 / 2`, `7 % 2` | `3`, `1` |
| `-7 / 2`, `-7 % 2` | `-3`, `-1` |
| `7 / -2`, `7 % -2` | `-3`, `1` |
| `-7 / -2`, `-7 % -2` | `3`, `-1` |

**This is not the numeric-mode rule, and the difference is deliberate.** In
numeric mode `%` is the floored modulo, `x - floor(x / y) * y`, so `-7 % 3` is
`2` there and `-1` here. Numeric mode wants the floored answer because a modulo
on tensor data is nearly always a wrap into a range, where a negative result is
a bug. Systems mode wants the truncating answer because the code that uses it is
digit extraction, quantisation and packing, all of which are ports of integer
code that assumes it. Two modes, two rules, one sentence each, rather than one
rule that is wrong for one of them. When a systems-mode program really wants the
floored answer, write `((a % b) + b) % b`.

**`shr` is not division.** `shr(a, k)` is `floor(a / 2^k)` and `/` truncates, so
the two agree for a non-negative `a` and disagree for a negative one:
`shr(-7, 1)` is `-4` and `-7 / 2` is `-3`. Replacing a division by a power of two
with a shift is a valid rewrite only when the dividend cannot be negative. This
is the sharpest edge in the integer half of the subset and it is the reason both
rules are written on the same page.

**Overflow.** `/` wraps like every other `I64` operation rather than trapping,
so `MIN_I64 / -1` is `MIN_I64` and `MIN_I64 % -1` is `0`. It is the only
division that overflows.

**Division by zero.** `a / 0` and `a % 0` abort the program with a diagnostic
naming the operation and its position. They do not return `0`, they do not
return a NaN, and they do not return a `Res`.

`docs/self-hosting.md` section 1.2 says "division by zero is an error value, not
a panic", and this narrows that sentence rather than restating it. What section
1.2 is ruling out is the Go bootstrap's behaviour, a host-language crash with a
stack trace and no source position, and that is ruled out. What it must not be
read as asking for is `/` returning `Res[I64, Str]`: that would put a `?` on
every arithmetic expression in the self-hosted compiler, and a language where
`i + 1` is fallible is a worse language than one where a zero divisor is a bug
that stops. A caller that expects a zero divisor tests for it, which is one line
at the one place it can happen rather than a type change at every place it
cannot.

## Equality

`==` and `!=` are **deep structural comparison**. Two values are equal when they
have the same type and the same contents, all the way down:

```rust
[1.0, 2.0] == [1.0, 2.0]                       # true (tensors)
[1.0, "x"] == [1.0, "x"]                       # true (lists, element by element)
{ w: [1.0], b: 0.5 } == { w: [1.0], b: 0.5 }   # true (records, field by field)
```

The details:

- A tensor's **shape is part of its value**: `[[1.0, 2.0], [3.0, 4.0]]` and
  `[1.0, 2.0, 3.0, 4.0]` hold the same numbers but are not equal. Numbers compare
  by IEEE rules, so a tensor holding a `NaN` is not equal to itself.
- Lists compare elementwise, and must be the same length.
- Records compare field by field, **matched by name**, so declaration order
  doesn't change the answer: `{ a: 1.0, b: 2.0 } == { b: 2.0, a: 1.0 }`. A record
  with an extra field is not equal.
- Values of different types are never equal: `[1.0] == 1.0` is false, not an
  error.
- Functions have no structure worth walking, so they compare by **identity**: a
  function equals itself, and two separately written `fn(x) = x` do not.
- `!=` is exactly the negation of `==`.

The ordering operators (`<`, `<=`, `>`, `>=`) are only defined on scalars;
applying one to a list, record, string, or non-scalar tensor is an error.

For elementwise comparison of two tensors into a tensor of 1s and 0s, use the
`equal` builtin. `==` on tensors gives one Bool for the whole value.

## Strings

A `Str` is an immutable byte string. It is bytes that print, not text: there is
no character type, no rune, and no unicode normalization anywhere in the
language. Everything below follows from that and holds in both modes.

### Equality

`==` on two `Str` values is true when they have the same length and the same
bytes, in order. There is no case folding, no whitespace trimming, no unicode
normalization, and no locale. Two strings that a person would call the same word
are equal only if they are the same bytes, so a decomposed and a precomposed
form of the same accented letter are different strings, and that is the answer
rather than a limitation: a lexer that compares source bytes against literals
needs byte equality and would be wrong under any other rule.

This is the general deep-equality rule from [Equality](#equality) applied to
`Str`, not a special case, so the surrounding rules come with it. A `Str` is
never equal to a value of another type, and that is `false` rather than an
error: `"1" == 1` is false. `!=` is exactly the negation.

`src/term/caps.tw` leans on this for every environment comparison, and
`docs/needs.md` NEEDS-46 records it as a constraint on the `Str` rewrite: making
`Str` a distinct indexable value must not change what `==` means.

### Ordering

**`<`, `<=`, `>` and `>=` are not defined on `Str`.** They remain scalar-only,
in both modes, and applying one to a string is an error. That is what the
bootstrap already does and what the existing code already assumes: `src/check.tw`
and `src/fmt.tw` both hand-write a byte-by-byte comparison rather than reaching
for `<`, as do spool's four sorts.

The ordering those hand-written comparisons implement is the one to keep, and it
is written down here so that eleven copies of it in six repositories agree:

> Compare byte by byte from index 0. At the first index where the two differ,
> the string with the smaller byte value is smaller. If one string runs out
> first and every byte matched, the shorter one is smaller. Bytes are compared
> as unsigned values in 0..255.

That is Go's `sort.Strings` and it is lexicographic on bytes, not on characters:
for ASCII it is alphabetical with uppercase before lowercase, and for anything
else it is UTF-8 code-point order, which is a well-defined order and not a
linguistic one. Any eventual `str_less` builtin or comparator means exactly
this.

The reason ordering is a function rather than an operator, when equality is an
operator, is that equality has one obvious meaning on bytes and ordering has
several plausible ones. `<` on a string would read as "alphabetically before",
which is a promise about language that a byte comparison does not keep. A named
function is a place to put that sentence.

### Concatenation

**`Str + Str` exists and produces a new `Str`.** It is the one overload of `+`
that is not numeric, and the result is the left operand's bytes followed by the
right operand's.

```rust
"col_" + name + "_0"
```

`+` on a `Str` and a non-`Str` is an error, with no coercion in either
direction. A number is converted with `str()` first, deliberately: an implicit
`str()` inside `+` would make `1 + 2` ambiguous the moment either side came from
a dictionary, and the explicit call is one function name in exchange for that.

`docs/self-hosting.md` gave `Bytes` a `concat` and gave `Str` length, indexing
and slicing, and never said how two strings join. Every codebase in the
ecosystem assumed `+` and spool calls it the single most-used operation in its
source, so this ratifies what was already assumed rather than overriding it.

**Building a string in a loop with `+` is quadratic.** Each `+` allocates and
copies the whole left operand, so `out = out + piece` across n pieces copies
O(n^2) bytes. This is stated here because it is the shape everything reaches for
first and it is affordable exactly until it is not: weft builds a frame of a live
plot from a few hundred pieces at 30 repaints a second, and twill's own
`src/cli/progress.tw` builds a bar a cell at a time. Use the `Bytes` builder for
those:

```rust
let out: Bytes = bytes.new()
bytes.push_text(out, piece)      # amortized O(1) per push
bytes.to_str(out)                # one copy, at the end
```

`src/bytes.tw` wraps that surface (`new`, `push`, `push_text`, `to_str`, plus
`concat`, `join` and `repeat`), and it is where a renderer's inner loop belongs.
`+` is for the outer one.

## Bindings and assignment

```rust
let x = 10     # new binding in the current scope
x = x + 1      # reassign an existing binding (error if not yet bound)
```

`let` always introduces a new variable. Plain assignment updates the nearest
existing binding, which is what makes training loops work.

## Functions

```rust
fn square(x) = x * x       # expression body
fn norm(v) {               # block body; last expression is returned
  let s = sum(v * v)
  sqrt(s)
}
let inc = fn(x) = x + 1    # anonymous function
```

Functions are values and close over their environment:

```rust
fn adder(n) = fn(x) = x + n
let add5 = adder(5)
add5(10)                   # 15
```

`return` exits early; a bare `return` yields `()`.

Parameters may carry shape annotations, and a function may declare its return
shape. These are checked statically (see below); at runtime they're ignored.

```rust
fn matvec(A: [3, 2], x: [2]) -> [3] {
  A @ x
}
```

A dimension can be a concrete size, `_` for an unknown, or a name (a shape
variable). A name used in more than one place must stand for the same size, so
the checker can tie shapes together and verify the return:

```rust
fn matmul2(A: [n, k], B: [k, m]) -> [n, m] {
  A @ B
}
```

Here `k` must match between `A` and `B`, and the result is checked against
`[n, m]`.

## Control flow

`if` is an expression:

```rust
let sign = if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
```

`while` and `for` are statements:

```rust
while i < n { i = i + 1 }

for k in range(5) { print(k) }      # over a list
for xi in [1.0, 2.0, 3.0] { ... }   # over a 1-D tensor
```

### `break` and `continue`

In systems mode a loop body may contain `break` and `continue`.

```rust
while i < len(src) {
  let c: I64 = src[i]
  if c == 32 { i = i + 1  continue }
  if c == 35 { break }
  push(toks, scan(lx))
}
```

- `break` leaves the innermost enclosing loop immediately.
- `continue` skips the rest of the body and begins the next iteration: in a
  `while` that means re-evaluating the condition, and in a `for` that means
  advancing to the next element. In neither case does it re-run anything already
  run in this iteration, so a `continue` in a `while` whose counter is advanced
  at the bottom of the body is an infinite loop, and that is the caller's bug and
  not a special case here.
- Both bind to the **innermost** enclosing loop. There are no labels and no
  multi-level break. A loop that wants to leave two levels sets a flag or is a
  function that returns.
- Both are statements, not expressions. A loop still evaluates to `()`, so
  neither carries a value and `break x` is a syntax error.
- Neither crosses a function boundary. A `fn` written inside a loop body is a
  new scope for this purpose, and a `break` in it is an error rather than a way
  to leave the loop that lexically encloses it. Use `return`.
- Both outside any loop are a checker error naming the statement.

**They are keywords in systems mode only**, at statement position, which follows
the rule `match` already uses. A numeric-mode file that writes `let break = 3`
keeps working, which is why the mode gate is worth the sentence: nothing in
`docs/language-guide.md`'s numeric half changes meaning.

Five parsers in the ecosystem were written against these and none of them could
point at a rule. `docs/needs.md` NEEDS-12 has the cost: twill's own scanner loop
is a chain of `continue`s and nests eight deep rewritten as nested `else`, and
bobbin's sampling loop carries a `done` flag for a loop with four exit
conditions.

## Indexing and slicing

```rust
let v = [10.0, 20.0, 30.0]
v[0]                  # 10 (scalar)

let m = [[1.0, 2.0], [3.0, 4.0]]
m[1]                  # tensor([3, 4], shape=[2]), a row
m[1][0]               # 3
```

Indexing a tensor along the first axis returns a scalar (rank-1) or a slice
(higher rank). Lists index directly.

Slicing takes a half-open range along the first axis; either bound may be
omitted. Both indexing (`x[i]`) and slicing (`x[a:b]`) a tensor are
differentiable: gradient flows to the selected element or rows.

```rust
v[1:3]                # tensor([20, 30], shape=[2])
v[:2]                 # first two elements
v[1:]                 # from index 1 to the end
m[0:1]                # the first row, kept as a [1, 2] tensor
range(10)[2:5]        # works on lists too
```

## Differentiation

```rust
grad(f)            # -> function returning df/d(arg0)
grads(f)           # -> function returning [df/d(arg0), df/d(arg1), ...]
value_and_grad(f)  # -> function returning [f(args), df/d(arg0)]
jacobian(f)        # -> function returning the [m, n] Jacobian of a vector output
hessian(f)         # -> function returning the [n, n] Hessian of a scalar output
```

`grad`, `grads`, and `value_and_grad` require the differentiated function to
return a scalar; a gradient has the same shape as the argument it corresponds to,
including nested lists. `jacobian(f)(x)` instead takes a function with a *vector*
output and returns the full matrix of partials, where row `i` is the gradient of
output `i`, computed by one reverse-mode pass per output. See
`examples/jacobian.tw`.

```rust
grad(fn(x) = x * x)(4.0)                 # 8
grad(fn(w) = sum(w * w))([1.0, 2.0])     # [2, 4]

let g = grads(fn(a, b) = sum(a * b))([1.0, 2.0], [3.0, 4.0])
g[0]   # [3, 4]   d/da
g[1]   # [1, 2]   d/db
```

Differentiable primitives: `+ - * / % @ ^`, `relu`, `sigmoid`, `tanh`, `exp`,
`log`, `sin`, `cos`, `sqrt`, `sum`, `mean`, `abs`, `pow`.

`hessian(f)(x)` gives the exact matrix of second partial derivatives of a scalar
function, by second-order autodiff via forward-mode jets (see `examples/hessian.tw`
for Newton's method). It supports functions built from arithmetic, the unary
math functions, `matmul`, `sum`, `mean`, and the structural ops indexing
(`x[i]`), slicing (`x[a:b]`), `reshape`, `transpose`, `concat`, and `gather`; a
function using an op outside this set raises a clear error. The reverse-mode `grad` remains
first-order, so the general nested form `grad(grad(f))` is not supported: it is
refused with an error naming `hessian`, rather than returning the zero it would
otherwise compute. The gradient `grad` hands back is a plain value with no
history, so differentiating it again differentiates a constant.

## Shape checking

`twill check` (and the check that runs before `twill run`) infers tensor shapes
and reports mismatches it can prove. It stays quiet when a shape can't be
determined, so dynamic code doesn't produce false alarms.

```
$ twill check bad.tw
bad.tw:3: shape error: shape mismatch in @: [2, 3] @ [2] (inner 3 != 2)
```

Annotations (`[3, 2]`, `[2]`, `[]`, `_` for unknown, or named shape variables)
let you state a contract that the checker enforces at call sites and against the
function body. A shape variable used more than once must resolve to the same
size. Annotated function bodies are also checked at their definition, so a
mistake is caught even if the function is never called.

## Units of measure

Declare a base unit at the top level with `unit`, then annotate scalar
quantities with it. The checker tracks units through arithmetic and reports a
mismatch, the same way it does for shapes, but units are erased at runtime, so
annotated code runs as plain numbers with zero overhead.

```rust
unit USD
unit share

fn notional(px: USD/share, qty: share) -> USD { px * qty }
```

An annotation is a single unit (`USD`) or a compound expression: a product
(`USD*share`), a quotient (`USD/year`), or a power (`year^-1`, `USD^2`). The
checker applies the natural rules:

- `*` multiplies units, `/` divides them, and `^` with a constant integer
  exponent raises them (`sqrt` halves them).
- `+`, `-`, `%`, and comparisons require both sides to share a unit. Adding
  `USD` to `share` is an error.
- `matmul`/`dot` multiply the operand units; indexing and slicing preserve them.
- `exp`, `log`, `sin`, `cos`, `tanh`, and `sigmoid` require a dimensionless
  argument (their result is dimensionless).

A bare numeric literal is dimensionless. To give a value a unit, annotate the
`let` that binds it: the literal is adopted into the declared unit:

```rust
let price: USD/share = 150.0
let qty: share = 200.0
let value = notional(price, qty)   # inferred: USD
```

Naming a unit that was never declared (a typo like `USD/yr`) is a checker error.
Code with no unit annotations is entirely dimensionless and unaffected.

## Records

A record groups named fields. Fields are accessed with `.`.

```rust
let p = { w: [1.0, 2.0], b: 0.5 }
p.w                   # tensor([1, 2], shape=[2])
p.b                   # 0.5
{ inner: { x: 3.0 } }.inner.x   # 3
```

`grad` follows record structure: if a loss takes a record of parameters, the
gradient is a record with the same fields.

```rust
fn loss(m) = sum(m.w) + m.b
grad(loss)({ w: [1.0, 2.0], b: 0.5 })   # { w: [1, 1], b: 1 }
```

This makes a record a natural container for a model's parameters. A `{` starts a
record only when it is followed by `name:`; otherwise it is a block.

A record type can be declared and used to annotate a parameter. The checker then
verifies that the record passed in has the declared fields with the declared
shapes, and that field accesses name real fields:

```rust
type Model = { w: [3, 2], b: [3] }

fn predict(m: Model, x: [2]) -> [3] {
  m.w @ x + m.b
}
```

Accessing a field a record doesn't have (`m.wieght`) is a checker error, whether
the record is a literal or a declared type.

## `struct`, and what a parameter is

`struct` is a systems-mode type, declared by name, with typed fields that are
mutable in place. It is a **different type from `Record`** and the two are not
unified, deliberately: `grad` walks a record's structure and depends on records
not aliasing, so mutation is not retrofitted onto them.

```rust
struct Lexer { src: Str, i: I64, line: I64 }
```

### The rule

**A `struct` has reference semantics. Passing one passes a handle, not a copy.
Assigning to a field of a parameter mutates the caller's struct, and the caller
sees it.** The same holds for `Arr`, `Dict` and `Bytes`, and it holds through
any number of levels: mutating a field of a struct reached through a field of
another struct is visible at the outermost handle.

```rust
fn advance(lx: Lexer) {
  lx.i = lx.i + 1
  if lx.src[lx.i] == 10 { lx.line = lx.line + 1 }
}

let lx: Lexer = Lexer { src: text, i: 0, line: 1 }
advance(lx)
# lx.i is 1 here, not 0.
```

Copying is always explicit. There is no implicit copy at a call, at an
assignment, at a `push` into an `Arr`, or at a return. `let b = a` on a struct
binds a second name to the one struct, and mutating through either is visible
through the other. A copy is made by writing one.

`F64`, `I64`, `Bool`, `Byte` and `Str` are the values that are not handles.
They are immutable, so the distinction is unobservable for them, which is the
point: the line between the two halves of the language is exactly the line
between mutable aggregates and immutable scalars, and there is nothing in
between to remember.

### Why this is written down

Three codebases were built on this and none of them could point at a rule.
`docs/self-hosting.md` section 1.2 says a struct has reference semantics, and
then says nothing about what happens when a function assigns to a field of a
parameter, and nothing at all about `Arr`. That is the gap this closes, and it
is the whole of it: nobody was asking for a feature.

The cost of leaving it open is not evenly distributed, which is why it is worth
a paragraph rather than a line. If the answer had been by-value, loom's `fit`
could not advance the run it was given and every function in loom's
`src/metrics.tw` would have to return a new meter, which is loud: nothing works
and it is obvious that nothing works. `src/tensor.tw`'s is the quiet one.
`accumulate(cot, touched, node, buf)` mutates `cot[node].data` and expects the
caller to see it, so if an `Arr` parameter were copied the mutation would be a
no-op, the backward pass would return zeros, and a gradient of zero is not an
error. It is a model that does not learn, and the search for the reason starts at
the learning rate.

This is also why the `Arr` half and the `struct` half must give the same answer
and are stated together. `Odometer` is a struct holding arrays and is mutated
through both at once; two different answers would make it work in one direction
and not the other.

### What the Go bootstrap does

The bootstrap agrees, as far as it can be asked. `internal/value`'s aggregates
are `*Record` and `*List`, Go pointers, and the interpreter passes them to a
call without copying, so aggregates are already handles there.

What the bootstrap does not have is any syntax that mutates one. Field
assignment (`p.b = 1.0`) is a parse error, `Arr` element assignment is
`docs/needs.md` NEEDS-43, and the only builtins that look like mutation are
`append` and `with_field`, both of which return a new value and leave the
original alone. So the reference-versus-value distinction is currently
unobservable from a twill program, and the rule above is a decision about
systems mode rather than a measurement of numeric mode. It is the decision every
existing caller already assumed, which is the evidence for it being the right
one.

`Record` in numeric mode keeps its own rule and is unaffected: fields are not
mutable in place, and you rebuild the record.

## Imports

There are two kinds of import path, and the spelling tells you which is which.

```rust
import "std/nn"             # a standard-library module (ships inside the binary)
import "helpers.tw"         # a file, relative to the importing file
```

A path beginning with `std/` names a **module** of the standard library, not a
file: no extension, no directory, and it means the same thing from anywhere,
because the library is compiled into the `twill` binary. `std/` is reserved, so a
directory called `std` next to your program does not shadow it. Every other path
is a **file**, resolved relative to the importing file first, then the working
directory; `import "./std/local.tw"` reaches a real directory named `std`.

Either kind can be namespaced:

```rust
import "std/nn"             # drops the module's definitions into this scope
import "std/nn" as nn       # binds them as a namespace record instead
```

A plain import evaluates the module and adds its top-level definitions to the
importing scope; each module loads once, so re-imports and cycles are fine. An
`as name` import instead evaluates it into its own scope and binds a record of
its definitions under `name`, so you call `nn.dense(...)`. That record's fields
are in the module's declaration order, so printing or iterating a namespace gives
the same result on every run.

A standard-library module may only import other `std/` modules. It has no
directory of its own to resolve a relative path against.

To work on the library itself without rebuilding, set `TWILL_STD` to a directory
of `.tw` files; it replaces the embedded library wholesale, so `import "std/nn"`
reads `$TWILL_STD/nn.tw`. Unset it and you are back to the copy in the binary.

## Standard library

Elementwise math (differentiable): `relu`, `sigmoid`, `tanh`, `exp`, `log`,
`sin`, `cos`, `sqrt`, `square`, `abs`, `pow(x, p)`, `clip(x, lo, hi)`.

Elementwise combine: `maximum(a, b)`, `minimum(a, b)`, `where(cond, a, b)`, and
the comparisons `greater`, `less`, `greater_equal`, `less_equal`, `equal`
(each returns a tensor of 1s and 0s).

Reductions: `sum`, `mean`, `max`, `min`, `prod` and `median` reduce the whole
tensor to a scalar, or one axis with a second argument (`sum(t, 0)`).
`argmax(t[, axis])` gives the index of the maximum.

All of them are differentiable, including the two order-based ones, though what
that means is worth being clear about. `median` routes the whole gradient to
whichever element was selected, the way `max` does, and splits it in half
between the middle two when the run has even length. `prod` gives each factor
the product of the others, which is the total divided by that factor, except
where a factor is zero and the division is not available. There, a single zero
takes the product of the rest and everything else gets nothing, and two or more
zeros flatten the gradient entirely, because every product of the others still
contains a zero. `softmax(t[, axis])` and `logsumexp(t[, axis])` default to
the last axis.

`split(t, n | sizes[, axis])` is the inverse of `concat`, returning a list of
pieces. A number means that many equal pieces (`split(x, 2, 1)` halves the
columns) and a list means those exact lengths (`split(x, list(1, 3), 1)`). The
axis defaults to 0. The sizes must account for the axis exactly and an equal
split must divide evenly; both are errors rather than ragged output, because a
split that quietly loses a row shows up later as a wrong loss rather than as a
crash. Each piece keeps its own gradient path, so
`concat(split(t, 2, 1), 1)` is `t` in both directions.

`broadcast_to(t, ...shape)` expands a tensor to a given shape under the usual
right-aligned rules, where every axis must already match or be 1. It is what
you need after a reduction: reducing axis 1 of a `[2, 3]` gives a `[2]`, and
`[2]` will not broadcast back against `[2, 3]`, because alignment is from the
right. `broadcast_to(reshape(mu, list(2, 1)), list(2, 3))` puts it back. Other
array libraries spell this as `keepdims=True` on the reduction itself; here it
is an operation, and `num.keep` wraps the two steps.

Sorting: `sort(t[, axis[, descending]])` and `argsort` give the values and the
positions; `topk(t, k[, axis[, smallest]])` and `argtopk` keep the k largest,
largest first, shrinking that axis to k. All four default to the last axis,
because sorting a matrix almost always means sorting each row. The flags are
numbers, since a comparison in Twill already yields 1 and 0.

`sort` and `topk` are differentiable and exactly so. Sorting is a permutation
and the derivative of a permutation is its inverse: whatever gradient arrives at
the element now in a position belongs to whichever element started there. A
value outside the top k does not move the output at all, so its gradient is
zero, which is correct rather than a simplification. The sort is stable, so ties
keep their original order and therefore their own gradients.

`argmin(t[, axis])` is `argmax`'s counterpart, and `flip(t[, axis])` reverses
along an axis. `flip` is differentiable and exactly so, since a reversal is a
permutation and is its own inverse, which makes the backward pass the same
reversal. All three default to the last axis. Ties in `argmax` and `argmin` go to
the first occurrence, the same rule the cumulative extremes and the sort use.

`roll(t, shift[, axis])` shifts along an axis and wraps what falls off the end
back to the start; `diff(t[, axis])` is the difference between neighbours,
shortening that axis by one. Both are differentiable. A positive shift moves
elements towards the end, so `roll(x, 1)` is the previous value and
`x - roll(x, 1)` compares a series with its own past. `diff` shortens rather than
pads, because there is no honest first difference: a zero there claims "no
change" about data that does not exist, and it is exactly the claim whatever
consumes the series next will believe.

`argsort` and `argtopk` are not differentiable, and not by omission: an index
does not move when an input moves slightly, then jumps when two values cross.
The derivative is zero almost everywhere and undefined on the boundaries.

Cumulative scans (preserving length): `cumsum`, `cumprod`, `cummax`, `cummin`.
These build signals, equity curves, and running peaks, and they are
differentiable: `cumsum` and `cumprod` have exact gradients (`cumprod` handles
zeros in the series), and `cummax`/`cummin` send each output's gradient to the
element the running extreme came from, ties going to the earlier one.

Each takes an optional axis: `cumsum(t)` scans the tensor's elements in order
and `cumsum(t, 1)` scans along axis 1, one run per row, keeping the shape. The
split follows the reductions, where `sum(t)` covers everything and `sum(t, 0)`
works per axis. On a 1-D tensor, which is what a sequence is, the two forms are
the same thing, so the axis is a widening rather than a second meaning.
Elementwise rounding `floor`, `ceil`, and `round` are forward-only (their
derivative is zero wherever it exists), handy for turning random draws into
integer ids.

Linear algebra / shape: `matmul(a, b)` / `dot(a, b)` (same as `@`),
`transpose(t[, ...axes])`, `reshape(t, ...shape)`, `concat(list, axis)`,
`einsum(spec, ...tensors)`.

Indexing / batching: `gather(x, indices)` selects rows of `x` (its first axis)
by an index list or 1-D tensor, and is differentiable: the gradient scatters
back to the selected rows, so repeated indices (embedding lookups) accumulate.
`permutation(n)` returns a seeded random ordering of `0..n-1` (for shuffling),
and `int(x)` truncates a scalar toward zero.

Convolutions (differentiable): `conv2d(input, weight)` is a 2-D cross-correlation
with `input` shaped `[Cin, H, W]` and `weight` shaped `[Cout, Cin, KH, KW]`,
producing `[Cout, H-KH+1, W-KW+1]` (valid padding, unit stride).
`maxpool2d(input, k)` does non-overlapping `k×k` max pooling over each channel of
a `[C, H, W]` tensor. `grad` flows through both, so a convolutional net trains
like any other model. See `examples/cnn.tw`.

`einsum` is a general Einstein-summation contraction and is differentiable:

```rust
einsum("ij,jk->ik", A, B)   # matrix multiply
einsum("ij->ji", A)         # transpose
einsum("ij->i", A)          # sum over the second axis
einsum("i,ij,j->", x, W, y) # bilinear form x' W y
```

Each label names an axis; repeated labels across operands are summed, and only
the labels in the output remain. Omitting `->` keeps the labels that appear once.
(A label repeated within one operand, a trace or diagonal, is not supported
yet.)

Construction: `tensor(list)`, `scalar(x)`, `zeros(...shape)`, `ones(...shape)`,
`fill(value, ...shape)`, `eye(n)`, `randn(...shape)` (standard normal),
`rand(...shape)` (uniform), `seed(n)`. Shapes may be separate args or a list.
Randomness is **deterministic by default**: a program gives the same result
every run, and `seed(n)` chooses the starting point. That reproducibility
matters for model governance and audit.

Lists / higher-order: `range(...)`, `list(...)`, `map(f, xs)`, `zip(...)`,
`fold(f, init, xs)`, `append(xs, x)`, `enumerate(xs)`, `len(x)`.

Trees (tensors nested in lists/records): `map_leaves(f, tree)` applies `f` to
every tensor leaf; `zip_leaves(f, trees)` walks a list of same-shaped trees in
parallel, calling `f` with the list of leaves at each position. Optimizers use
these, so they work on any model structure.

Inspection: `shape(t)`, `item(t)`, `str(x)`, `print(...)`.

### `str` on a number

**`str(n)` for an `I64` is the digits and nothing else: no decimal point, no
trailing `.0`, no exponent, no thousands separator, no padding, and a leading
`-` exactly when the value is negative.** `str(0)` is `"0"`. `str(-7)` is
`"-7"`. The most negative `I64` renders in full as
`"-9223372036854775808"`, which is the one value whose negation is not
representable and therefore the one a digit loop gets wrong.

This is stated because `str` on a scalar goes through the tensor printer, and a
`.0` from that path would land in every line number, every column count and
every axis index in every diagnostic the self-hosted compiler emits. It is not a
formatting preference, it is the difference between `lex.tw:294:12` and
`lex.tw:294.0:12.0`.

`str(x)` for an `F64` prints a whole number with no decimal point and everything
else in Go's shortest round-tripping `%g`, which is what the bootstrap's
`internal/value.FormatNumber` does and what `src/float.tw`'s `format_number`
reimplements. The switch to exponent form happens at decimal exponent below -4
or at or above 6, so `1000000` prints as `1000000` and `1234567.5` prints as
`1.2345675e+06`.

The two rules meet where an `F64` holds an integral value, and `format_number`
routes that case to the `I64` renderer on purpose, so a whole number prints the
same however it was computed.

`str` does no padding and no alignment. `str(7)` is `"7"` and never `"  7"`.
Column alignment is a caller's job until a formatting builtin exists
(`docs/needs.md` NEEDS-20), and bobbin's `pad_left` and `pad_right` are what
that looks like meanwhile.

A caveat worth knowing before it is discovered: in **numeric mode** there is no
`I64`, so an integer is a float64 and `str` is exact only up to 2^53.
`str(123456789012345678)` prints `123456789012345680`. Systems mode is where an
integer is an integer, and this is one of the reasons it exists.

Data: `read_csv(path)` loads a file of numeric rows (comma- or
whitespace-separated, `#` lines skipped) into a `[rows, cols]` tensor.

Persistence: `save(value, path)` writes any value, whether a tensor, a record or list
of tensors (a model's whole parameter tree), or a fitted `gbm` model, to a file
in an exact binary format, and `load(path)` reads it back. Paths are relative to
the running script. This is the deploy path: train once, `save` the model, and
ship it with the single binary for inference (`examples/save_load.tw`).

Frames: a frame is a record whose fields are named column tensors, so field
access, slicing, and `grad` all work on it. `read_frame(path)` loads a CSV whose
first row is a header into such a record; `write_frame(frame, path)` writes one
back. `columns(rec)` lists the field names, `field(rec, name)` looks one up by
string, and `with_field(rec, name, value)` returns a copy with a field set. See
`examples/frames.tw`.

Gradient-boosted trees: `gbm_fit(X, y)` (or `gbm_fit(X, y, opts)`) trains a
native gradient-boosting model on a `[n, d]` feature matrix and an `[n]`
target/label vector, and `gbm_predict(model, X)` scores a `[n, d]` matrix into an
`[n]` vector. `opts` is a record of hyperparameters: `rounds`, `learning_rate`,
`max_depth`, `min_leaf`, `lambda`, `gamma`, and `objective` (`"squared"` for
regression, `"logistic"` for binary classification, where predictions are
probabilities). The engine is pure Go and deterministic. See `examples/gbm.tw`.

Libraries written in Twill ship inside the binary and are imported as
`std/<module>`: `std/nn` (layers including `dense`, `conv`, `embed`, and
`self_attention`; activations, initializers, losses), `std/optim` (SGD,
momentum, Adam), `std/data` (`standardize`, `train_test_split`, `shuffle` for
real training loops, see `examples/minibatch.tw`), and `std/backtest`
(returns, moving averages, equity curves, drawdown, Sharpe, Sortino, volatility,
CAGR). Their sources are the `.tw` files in `std/` in the repository. The
optimizers are container-agnostic: the same `sgd_step`/`adam_step`
update a model held in a positional list or a named record. The backtest Sharpe
and Sortino are differentiable in the return series, so a smooth signal can be
tuned by gradient ascent through the backtest (`examples/signal_opt.tw`).

## Example

```rust
# Fit y = X w + b by gradient descent.
let X = [[1.0, 1.0], [2.0, 1.0], [1.0, 3.0]]
let y = [-0.5, 1.5, -6.5]

fn loss(w, b) {
  let err = X @ w + b - y
  mean(err * err)
}

let w = [0.0, 0.0]
let b = 0.0
for step in range(400) {
  let g = grads(loss)(w, b)
  w = w - g[0] * 0.05
  b = b - g[1] * 0.05
}
print("w =", w, "b =", b)
```
