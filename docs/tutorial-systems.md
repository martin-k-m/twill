# The systems half of Twill

Twill has two halves. The one in [the tutorial](tutorial.md) is for
mathematics: a scalar is a rank-0 tensor, `grad` differentiates whatever you
wrote, and shapes are checked before the program runs. This one is for writing
the programs that do everything else -- a lexer, a package manager, a
serialiser, twill's own compiler -- where a number is a machine word, a string
is bytes, and a data structure is something you mutate.

A file opts in on its first line:

```rust
mode systems
```

That line changes what the language is, and the rest of this document is what
it changes. Nothing here affects a numeric-mode file, which is the point of
having the line at all.

## Numbers are two types, and neither is a tensor

```rust
mode systems

let n: I64 = 42          # a 64-bit integer, exactly
let x: F64 = 0.5         # a float, held by value
```

An `I64` is two's complement and exactly 64 bits. It wraps rather than
trapping, `/` truncates toward zero, and `%` takes the sign of the dividend:

```rust
let a: I64 = -7
let b: I64 = 2
print(a / b, a % b)      # -3 -1
let c: I64 = 7
let d: I64 = -2
print(c / d, c % d)      # -3 1
```

Both operands have to be integers for that to be integer arithmetic, and a bare
literal is not one: `7 / 2` written out is `3.5` even here, because nothing said
`I64` about either number. The two ways to say it are an annotation, which
converts at the binding, and `//`, which truncates whatever it is given:

```rust
let mid: I64 = (lo + hi) / 2    # truncated at the binding
let half = 7 // 2               # 3, said at the operator
```

**That is not numeric mode's rule.** There `%` is the floored modulo, so
`-7 % 2` is `1`, because a modulo on tensor data is nearly always a wrap into a
range where a negative answer is a bug. Here it is digit extraction and
packing, which are ports of integer code that assumes truncation. Two modes,
two rules, and this is the sentence that says so.

`f64(n)` widens an integer, losing precision above 2^53; `i64(x)` narrows a
float, truncating toward zero. There is no implicit conversion in either
direction, ever. An `F64` is a machine word, not a rank-0 tensor: a running
total in a training loop would otherwise allocate a tensor and a tape node per
step, and the accumulator would cost more than the model.

The bitwise words are `band`, `bor`, `xor`, `bnot`, `shl` and `shr`, and they
are exact on all 64 bits including the sign:

```rust
print(shl(1, 63))        # -9223372036854775808
print(shr(-8, 1))        # -4, an arithmetic shift
```

`and` and `or` infix are the *boolean* operators and return an operand, so
`x and 255` is `255` for any non-zero `x` -- a silent wrong answer. The bitwise
meaning has its own name for exactly that reason.

## Text is bytes

```rust
let s: Str = "hello"
print(len(s))            # 5, in bytes
print(s[0])              # 104, the byte
print(s[1:3])            # "el"
print(s + " world")
```

A `Str` is immutable and is bytes that print: there is no character type, no
rune, and no normalisation anywhere. A lexer comparing source bytes against
literals needs exactly that and would be wrong under any other rule.

Building a string with `+` in a loop is quadratic, because each `+` copies the
whole left side. A `Bytes` is the growable one, and is what a renderer's inner
loop should use:

```rust
let out: Bytes = bytes_new()
bytes_push(out, 65)
print(bytes_to_str(out))
```

## Arrays, dictionaries, and structs, which are handles

```rust
let xs: Arr[I64] = arr_new()
push(xs, 10)
push(xs, 20)
xs[0] = 11
print(len(xs), xs[0], pop(xs))

let counts: Dict[Str, I64] = {}
dict_set(counts, "a", 1)
print(dict_has(counts, "a"), dict_get(counts, "zz"))
```

`dict_get` returns an `Opt`, so a missing key is a value you handle and not a
zero you fail to notice.

A `struct` is nominal, its fields are typed, and it is mutable in place:

```rust
struct Lexer { src: Str, i: I64, line: I64 }

fn advance(lx: Lexer) {
  lx.i = lx.i + 1
  if lx.i < len(lx.src) and lx.src[lx.i] == 10 { lx.line = lx.line + 1 }
}

let lx: Lexer = Lexer { src: "ab\nc", i: 0, line: 1 }
advance(lx)
advance(lx)
advance(lx)
print(lx.i, lx.line)     # 3 2
```

**Passing a struct passes a handle.** Assigning to a field of a parameter
mutates the caller's struct, and the caller sees it. The same holds for `Arr`,
`Dict` and `Bytes`, and through any number of levels. Copying is always
explicit: `let b = a` binds a second name to the one struct.

That is deliberately not what a numeric-mode `Record` does. `grad` walks a
record's structure and is correct only because records do not alias, so
mutation is not retrofitted onto them.

## Enums, and match

```rust
enum Tok { Ident(Str), Num(F64), Punct(Str), Eof }

fn describe(t: Tok) -> Str {
  match t {
    Ident(name) => "identifier " + name,
    Num(v) => "number",
    Punct(p) => p,
    Eof => "end of input",
  }
}
```

A case carries one value or none. **A `match` must cover every case**, and the
checker names the ones you left out:

```
lexer.tw:14: shape error: match on Tok is not exhaustive: missing Eof
```

That is the reason to have an enum rather than an integer tag. Adding a case to
the declaration makes every `match` that has not been updated say so, at check
time, instead of one of them falling through in a month.

## Failing

```rust
enum Opt[T] { Some(T), None }
enum Res[T, E] { Ok(T), Err(E) }
```

Both are ordinary enums. What is not ordinary is `?`, which unwraps a success
and returns a failure from the enclosing function:

```rust
fn load(path: Str) -> Res[Str, Str] {
  let text: Str = read_file(path)?
  Ok(text + "\n")
}
```

The enclosing function must return a `Res` or an `Opt`, which the checker
enforces, along with what `?` yields: `let n: I64 = read_file(p)?` is reported,
because `read_file` gives back a `Str`.

`abort(msg)` is the other way to fail and means something else entirely. A
`Res` is what a caller may reasonably handle -- a missing file, a malformed
line. `abort` is for an invariant broken inside your own implementation, which
no caller can do anything about. Every `abort` in twill's compiler should be
unreachable by any input.

## Files

```rust
match read_file("config.toml") {
  Ok(text) => print(len(text)),
  Err(e) => print("cannot read it:", e),
}

let dir: Str = match temp_dir("demo") { Ok(p) => p, Err(e) => abort(e) }
let f: Str = path_join(dir, "out.txt")
match write_file(f, "hello") { Ok(_) => unit, Err(e) => print(e) }
print(path_exists(f), path_base(f), path_ext(f))
match remove_all(dir) { Ok(_) => unit, Err(e) => print(e) }
```

`path_join` and its neighbours are string handling and touch nothing. They emit
a forward slash on every platform, because a program's paths are written in its
source and one that renders them differently on Windows writes a different file
there.

## Putting it together: a tiny expression parser

Everything above, in one program: an enum for the syntax tree, a struct for the
parser's position, a `match` that has to be exhaustive, and a `Res` for the one
thing that can go wrong.

```rust
mode systems

enum Expr { Num(F64), Add(Pair), Mul(Pair) }
struct Pair { lhs: Expr, rhs: Expr }
struct P { src: Str, i: I64 }

fn peek(p: P) -> I64 {
  if p.i >= len(p.src) { return 0 }
  p.src[p.i]
}

fn skip_spaces(p: P) {
  while peek(p) == 32 { p.i = p.i + 1 }
}

# A number: one or more digits, no sign and no decimal point.
fn parse_num(p: P) -> Res[Expr, Str] {
  skip_spaces(p)
  let start: I64 = p.i
  while peek(p) >= 48 and peek(p) <= 57 { p.i = p.i + 1 }
  if p.i == start { return Err("expected a digit at " + str(p.i)) }
  match str_to_f64(slice(p.src, start, p.i)) {
    Some(v) => Ok(Num(v)),
    None => Err("not a number"),
  }
}

# A product: numbers joined by `*`, which binds tighter than `+`.
fn parse_mul(p: P) -> Res[Expr, Str] {
  let left: Expr = parse_num(p)?
  skip_spaces(p)
  while peek(p) == 42 {
    p.i = p.i + 1
    let right: Expr = parse_num(p)?
    left = Mul(Pair { lhs: left, rhs: right })
    skip_spaces(p)
  }
  Ok(left)
}

# A sum: products joined by `+`.
fn parse_expr(p: P) -> Res[Expr, Str] {
  let left: Expr = parse_mul(p)?
  skip_spaces(p)
  while peek(p) == 43 {
    p.i = p.i + 1
    let right: Expr = parse_mul(p)?
    left = Add(Pair { lhs: left, rhs: right })
    skip_spaces(p)
  }
  Ok(left)
}

fn eval(e: Expr) -> F64 {
  match e {
    Num(v) => v,
    Add(pr) => eval(pr.lhs) + eval(pr.rhs),
    Mul(pr) => eval(pr.lhs) * eval(pr.rhs),
  }
}

fn run(src: Str) {
  let p: P = P { src: src, i: 0 }
  match parse_expr(p) {
    Ok(e) => print(src, "=", eval(e)),
    Err(msg) => print(src, "->", msg),
  }
}

fn main() {
  run("2 + 3 * 4")
  run("10 * 10 + 1")
  run("2 + ")
}
```

```
2 + 3 * 4 = 14
10 * 10 + 1 = 101
2 +  -> expected a digit at 4
```

Four things in that program are worth naming, because each is a decision the
language made rather than a convenience:

- `p` is mutated by `skip_spaces` and by every `parse_` function, and the caller
  sees it. That is what makes a recursive-descent parser writable without
  threading an index through every return.
- `eval`'s `match` has no `_`. Add a case to `Expr` and the checker reports
  `eval` immediately; a `_` there would have made it return a plausible number
  for a node it does not understand.
- `parse_mul` propagates its failure with one character. Written out, that
  function is three `match` statements deep and its actual work is at the
  bottom of them.
- `str_to_f64` returns an `Opt`, so the one place a conversion can fail is the
  one place it is handled.

The [language guide](language-guide.md) is the complete reference for both
halves, and `src/` in this repository is the largest systems-mode program there
is: twill's own lexer, parser, checker, formatter and evaluator, written in
twill.
