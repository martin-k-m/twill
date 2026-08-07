# What the self-hosted compiler needs from the language

The source under `src/` is twill's compiler written in twill. It does not run
today. This file is the reason it does not, item by item: every language or
runtime feature `src/` uses that the bootstrap implementation in `internal/`
does not yet provide.

It is the work queue for the systems subset. Each entry says what the feature
is, which file and line reaches for it, and what the Go bootstrap does in the
same place, so that implementing an entry is a matter of making the Go side do
in twill what it already does in Go.

Ordering is by dependency, not by priority: nothing below can be attempted
before the entries it rests on. The `NEEDS-n` ids are referenced from comments
in `src/`.

Status key: **blocking** means no file in `src/` parses without it.

---

## NEEDS-1 — `mode systems` as a file-level declaration

**Status:** blocking. Every file in `src/` opens with it.

The gate from `docs/self-hosting.md` section 1.1. The first non-comment line of
a file is `mode systems`; the default with no declaration is numeric mode and is
unchanged. The lexer must produce it as an ordinary keyword-plus-identifier
pair, the parser must accept it only as the first statement, and everything
below is refused outside it.

*Go bootstrap:* no concept of a mode. `internal/parser/parser.go` `parseStmt`
has no case for it and `internal/checker/checker.go` has a single policy.

## NEEDS-2 — `I64` with bitwise operators

**Status:** blocking. `src/lex.tw` uses it for every offset, line and column.

A signed 64-bit integer distinct from float64, with `and or xor shl shr not`,
defined wrapping, and explicit `i64()` / `f64()` conversions only.
`src/lex.tw:131` (`is_utf8_continuation`) and `src/lex.tw:498` (`utf8_width`)
mask lead bytes with `and`, which is the whole reason the subset needs an
integer type rather than a float that happens to hold integers.

*Go bootstrap:* every numeric value is `*tensor.Tensor`. `internal/interp` has
no integer type and no bitwise operator; `builtins.go` `int` truncates a float
and returns a float.

## NEEDS-3 — `enum` with payloads and exhaustive `match`

**Status:** blocking for `src/ast.tw` and everything downstream of it.

Tagged unions, matched by pattern, exhaustiveness enforced as an error. The AST
is forty variants and the token kind is seven; `src/lex.tw:29` spells the token
kinds as `I64` constants because milestone 1 excludes sum types, and the
`kind_name` ladder immediately below it is what that costs. `src/ast.tw` does
not attempt the workaround: an AST as integer tags plus parallel arrays is not
a design, it is a transcription error waiting to happen.

*Go bootstrap:* Go interfaces plus a type switch. `internal/ast/ast.go` declares
`Node`, `Stmt` and `Expr` as interfaces with unexported marker methods, and
every consumer type-switches. There is no exhaustiveness check anywhere, which
is precisely the property the twill version is meant to gain.

## NEEDS-4 — generics: `Arr[T]`, `Dict[K,V]`, `Opt[T]`, `Res[T,E]`

**Status:** blocking.

Type parameters on structs, enums and functions. No bounds, no traits,
monomorphized. Used on nearly every line of `src/`; `src/lex.tw:198` (`Lexer`
holds `Arr[Token]` and `Arr[Comment]`) is the first.

*Go bootstrap:* Go generics for none of it. `internal/value` has `List` as
`[]Value` (heterogeneous) and `Record` as ordered string keys, and the checker
models them as `tList` and `tRecord` with no element type.

## NEEDS-5 — `struct`: nominal, mutable, reference semantics

**Status:** blocking. `src/lex.tw:198` `Lexer` is a cursor that advances.

Fields typed, mutable in place, passed by handle. `advance(lx)` at
`src/lex.tw:240` mutates `lx.i`, `lx.line` and `lx.col` and the caller sees it.
Threading those three through every scan function and returning updated copies
would roughly double the lexer and make its diffs unreadable.

Must stay a distinct type from `Record`: `grad` over a record depends on records
not aliasing (`docs/self-hosting.md` section 1.2).

*Go bootstrap:* `value.Record` exists but is value-ish and has no declared
mutability. `interp` supports record field assignment only through rebinding.

## NEEDS-6 — indexable, sliceable `Str`

**Status:** blocking.

`s[i]` yields the `I64` byte value, `s[a:b]` yields a `Str` copy, `len(s)` is
the byte length. `src/lex.tw` is built on this: `scan_ident` at
`src/lex.tw:378` returns `lx.src[start:lx.i]` rather than accumulating, which
is both faster and the only way the token value is guaranteed byte-identical to
the source span.

*Go bootstrap:* `value.Str` is a Go string with no index, no slice, no length
and no concatenation. A lexer written in twill today cannot read its first
character.

## NEEDS-7 — `Bytes`: a growable byte buffer

**Status:** blocking. The whole of `src/bytes.tw`.

`bytes_new`, `bytes_push`, `bytes_to_str`. Everything the compiler prints is
built by appending. `src/bytes.tw:41` (`concat`) exists so that the rest of the
compiler never builds a string by repeated `+`, which is quadratic.

*Go bootstrap:* `strings.Builder`, used exactly this way in
`internal/lexer/lexer.go` and `internal/format/format.go`.

## NEEDS-8 — `Dict[Str, V]` with insertion-ordered iteration

**Status:** blocking.

`dict_new`, `dict_set`, `dict_get` returning `Opt[V]`, `dict_has`, `dict_del`,
`len`, and iteration in insertion order. `src/lex.tw:149` (`keyword_table`) is
the smallest use; `src/check.tw` uses it for the environment, the declared
types and the unit table.

Insertion order is not a nicety. `docs/self-hosting.md` section 2.4 makes
`compilerB == compilerC` the bootstrap check, and a symbol table iterated in a
random order emits different output on every run.

*Go bootstrap:* Go maps, whose iteration order is deliberately randomized.
`internal/checker/checker.go` gets away with it because it sorts before
printing (`unitString` calls `sort.Strings`) and because `Record` carries its
own `Keys` slice for exactly this reason.

## NEEDS-9 — a byte literal

**Status:** not blocking. Worked around in `src/lex.tw:91`.

There is no character literal, so `src/lex.tw` defines `ch(s) = s[0]` and writes
`let C_QUOTE = ch("\"")`. That is readable and it is a runtime call and a
one-byte allocation per constant at module load. A `b'x'` literal folding to a
constant would remove both. Low priority: the workaround is honest and the cost
is paid once.

*Go bootstrap:* Go rune literals, `'\n'` and friends, throughout the lexer.

## NEEDS-10 — `Res[T, E]` and postfix `?`

**Status:** blocking.

`src/lex.tw:294` (`tokenize`) is `Res[Arr[Token], SyntaxError]` and uses `?` to
propagate. Without it every call site of every scanner function grows an
explicit error check, and the checker in `src/check.tw`, which threads
diagnostics through about ninety functions, becomes unreadable.

The enclosing function must itself return a `Res`, checked statically.

*Go bootstrap:* Go `error` returns and `if err != nil`. `lexer.SyntaxError` is
the error value; `src/lex.tw:67` is its twill equivalent and matches its
`Error()` rendering exactly, so a message printed by either implementation is
the same string.

## NEEDS-11 — `abort(msg)`

**Status:** blocking. `src/lex.tw:45` and about thirty other places.

A panic that means a compiler bug, never a user error. Every `abort` in `src/`
should be unreachable for any input; that is a review rule and it is worth
keeping.

*Go bootstrap:* Go `panic`, used sparingly; most impossible cases in
`internal/interp` return an error instead, which conflates them with user
errors.

## NEEDS-12 — `continue` in `while` and `for`

**Status:** blocking. `src/lex.tw:305` and the whole scanner loop.

The Go lexer's main loop is a chain of `continue`s and the twill port is the
same shape. Rewriting it as nested `else` would nest eight deep.

*Go bootstrap:* Go `continue`. `internal/interp/interp.go` implements `for` and
`while` with no loop-control statements at all; the language has neither
`break` nor `continue`.

## NEEDS-13 — `Unit` as a value, and `unit` as its literal

**Status:** not blocking; cosmetic. `src/lex.tw:421`.

`scan_string` returns `Res[Unit, SyntaxError]`, so it needs something to put in
the `Ok`. The checker already has a `tUnit` type; the value side is missing.

*Go bootstrap:* `value.Unit`, which exists (`internal/value/value.go`) but has
no source syntax.

## NEEDS-14 — a `Bool` type name in annotations

**Status:** blocking, trivially.

`src/lex.tw:61` annotates a struct field `trailing: Bool`. The parser currently
reads a bare name after `:` as a record type or a unit
(`internal/parser/parser.go` `parseParam`), so `Bool` would be resolved as a
unit and reported as undeclared.

*Go bootstrap:* `checker.tBool` exists; there is no way to write it.

## NEEDS-15 — the whole-file lexer divergence: non-ASCII whitespace in comments

**Status:** known, accepted divergence. `src/lex.tw:466` (`is_space`).

Go's `strings.TrimSpace` trims a Unicode space set that includes U+0085 and
U+00A0. `src/lex.tw` trims the ASCII members only, because matching the rest
needs a UTF-8 decoder in the scanner for a case that cannot occur in a comment
without the file already being unusual. If the differential harness ever trips
on it, the fix is a decoder in `trim_space`, not in the scanner.

---

## Parser

## NEEDS-16 — recursive enum payloads without explicit indirection

**Status:** blocking. `src/ast.tw` throughout.

`Expr` holds `Expr`. The subset's answer is that an enum payload is a struct and
a struct is a handle, so the recursion needs no `Box`. That has to actually be
true in the implementation, including for a payload that is the enum itself
(`Unary { operand: Expr }`).

*Go bootstrap:* interfaces are already references.

## NEEDS-17 — a growable `Arr` with `pop` and `set`

**Status:** blocking. `src/parse.tw` builds every statement list this way.

`append(xs, x)` returns a new list today, so building an n-element list is
quadratic. `docs/self-hosting.md` section 1.2 makes the same argument for the
token stream.

*Go bootstrap:* Go slices with `append`.

## NEEDS-18 — `f64_of_str` / `str_to_f64`

**Status:** blocking. `src/parse.tw` `parse_number`.

The parser turns a NUMBER token's text into a float and must produce bit-exact
agreement with Go's `strconv.ParseFloat`, or two implementations disagree on
`0.1` and every downstream canonical dump differs. Correct decimal-to-binary
rounding is not something to reimplement in twill: this must be a runtime
primitive that calls the same conversion the Go side does.

*Go bootstrap:* `strconv.ParseFloat(t.Value, 64)` in
`internal/parser/parser.go` `parsePrimary`.

## NEEDS-19 — `i64_of_str`

**Status:** blocking. `src/parse.tw` shape dimensions and unit exponents.

The integer equivalent, matching `strconv.Atoi` including its overflow
behaviour, since `parseDim` reports "shape dimension must be a non-negative
integer" on a failure and the two implementations must fail on the same inputs.

*Go bootstrap:* `strconv.Atoi`.

## NEEDS-20 — string formatting

**Status:** blocking. Every diagnostic in `src/parse.tw` and `src/check.tw`.

`internal/checker` builds messages with `fmt.Sprintf` and the twill versions
must produce byte-identical strings, because the diagnostics are compared. The
subset needs at minimum the `%d`, `%s` and `%q` equivalents. `src/bytes.tw`
supplies the joining; what is missing is the rendering of a float the way Go's
`%g` does inside `str()`, and the Go-compatible quoting of a string, which
`src/lex.tw:478` (`quote_char`) approximates for one message only.

*Go bootstrap:* `fmt.Sprintf` throughout.

---

## Checker

## NEEDS-21 — identity of a heap value (`is_same`)

**Status:** blocking. `src/check.tw` recursion guard.

`internal/checker/checker.go` keys its recursion guard on `map[ast.Node]bool`,
that is, on the *identity* of an AST node, not its contents. Two structurally
identical lambdas in one file are different nodes and must not share a guard
entry. The twill version needs either pointer identity as a `Dict` key or a
unique node id assigned at parse time.

`src/check.tw` takes the second route and gives every AST node an `id: I64` at
construction, because identity-as-a-map-key is a language feature with
consequences (it leaks the collector's behaviour into the semantics) and a
serial number is not. The cost is one field on every node, and it also gives
the canonical dump a stable name for a node, so it is not purely a workaround.

*Go bootstrap:* pointer keys in a Go map.

## NEEDS-22 — `Opt[T]` returned from `Dict` lookup, and `match` on it

**Status:** blocking. Every environment lookup in `src/check.tw`.

Go's `v, ok := m[k]` is two returns; twill has one. `Opt` is the whole reason
`Res`/`Opt` are in the subset.

*Go bootstrap:* the comma-ok form.

## NEEDS-23 — sorting a `Arr[Str]`

**Status:** blocking. `src/check.tw` `unit_string`.

`internal/checker/checker.go` `unitString` calls `sort.Strings` on the unit's
base names before joining, so that `USD*year^-1` renders the same regardless of
map order. The twill version must sort identically (bytewise ascending) or unit
diagnostics differ between implementations.

`src/check.tw` implements an insertion sort rather than asking for a builtin:
the lists are two or three elements, and a `sort` builtin over `Arr[Str]` is a
comparator question that the subset does not need to answer yet.

*Go bootstrap:* `sort.Strings`.

## NEEDS-24 — integer division and modulo on `I64`

**Status:** blocking. `src/check.tw` `unit_sqrt` (`v % 2`, `v / 2`).

Float division would give `1.5` where the checker needs a failure. Defined
behaviour on division by zero as an error value, per
`docs/self-hosting.md` section 1.2.

*Go bootstrap:* Go's `/` and `%` on `int`.

---

## Evaluator and tensors

## NEEDS-25 — a foreign call into the native tensor core

**Status:** blocking for `src/eval.tw`, and by design.

`docs/self-hosting.md` section 2.2 draws the line: everything that reads twill
source is twill, everything that executes it is not. `src/tensor.tw` therefore
describes the tensor semantics and the autodiff tape, and calls primitives the
native core provides for the actual arithmetic. What is missing is the calling
convention: how a twill function names a core primitive, and what the checker
believes about its type.

Without a decision here, `src/tensor.tw` is a specification of behaviour with no
route to being executed, which is what it currently is.

*Go bootstrap:* `internal/interp/builtins.go` dispatches on a name into Go
functions in `internal/tensor`.

## NEEDS-26 — closures capturing a mutable environment

**Status:** blocking. `src/eval.tw` function values.

`internal/interp` closes over a `*Env` with a parent pointer and assignment
walks up to the defining scope. Twill closures exist; what is unspecified is
whether a captured variable is captured by handle or by value, and the
evaluator needs by handle to reproduce the bootstrap's behaviour for
`for i in ... { fns = append(fns, fn() = i) }`.

*Go bootstrap:* `interp.Env` with `assign` walking parents
(`internal/interp/interp.go`).

## NEEDS-27 — deep equality with the "different types are never equal" rule

**Status:** blocking. `src/eval.tw` `==`.

The bootstrap's rule, tested in `internal/interp/equality_test.go`: values of
different types compare unequal rather than raising. The twill version must
match it including the new subset types (I64 to I64 by bits, never equal to a
tensor).

*Go bootstrap:* `interp.valuesEqual`.

## NEEDS-28 — `read_file`, `write_file`, `args`, `exit`, `write_out`

**Status:** blocking. `src/main.tw`.

`read_file(path) -> Res[Bytes, Str]` and the rest of the process interface from
`docs/self-hosting.md` section 1.2. `src/main.tw` is a CLI and cannot exist
without them.

This is the only entry in this file that widens what an arbitrary `.tw` file can
do, and it should be landed knowing that.

*Go bootstrap:* `os.ReadFile` and friends in `cmd/twill/`.

## NEEDS-29 — a stable canonical rendering of a float

**Status:** blocking for `twill dump`, and the highest-risk item here.

The canonical dump in `testdata/` is compared byte for byte. Whatever formats a
float in `src/eval.tw` must agree with the Go side exactly, digit for digit,
including the shortest-representation rule. This is not a formatting preference,
it is the acceptance criterion of the whole differential harness.

Treat it as a runtime primitive that calls the same code the Go side calls.
Reimplementing Ryu or Grisu in twill to get a byte-identical answer is a way to
lose a month.

*Go bootstrap:* `strconv.FormatFloat` reached through the dumper in
`cmd/twill/dump.go`.

## NEEDS-30 — recursion depth, and a guard on it

**Status:** not blocking; an operational risk.

The parser and the checker are recursive descent over user input and the
bootstrap has the same exposure. The Go side survives on the goroutine stack;
what the twill side does on a 10,000-deep nesting is undefined until the VM
exists. Worth a depth counter with a diagnostic rather than whatever the VM
does by default.

*Go bootstrap:* none. A deeply nested twill file crashes the Go parser today.

## NEEDS-31 — deliberate divergence: `t[]`

**Status:** decided. `src/parse.tw` `parse_index_or_slice`.

`t[]` has no start expression and no `:` to make it a slice.
`internal/parser/parser.go` builds an `ast.Index` with a nil `Index` field and
the failure surfaces later in the evaluator, pointing at the wrong place.
`src/parse.tw` refuses it at the bracket with "expected an index expression".

This is the one place the twill parser is knowingly not a copy. It is recorded
here rather than silently fixed because the differential harness will report it
as a divergence, and a divergence with no note is indistinguishable from a bug.
Either the Go parser is changed to match, or this entry is the reason the diff
is expected.
