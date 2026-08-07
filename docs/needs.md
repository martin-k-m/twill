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

An id is permanent once assigned. Three agents appended concurrently and
collided on 68 through 72; the collision was resolved by moving one set, and
every comment in `src/` that named an id had to be re-read to find out which
entry it meant. A new entry takes the next number above the highest one in this
file and never reuses one, so a `NEEDS-n` in a comment means the same thing a
month later as it did when it was written.

---

## NEEDS-1: `mode systems` as a file-level declaration

**Status:** blocking. Every file in `src/` opens with it.

The gate from `docs/self-hosting.md` section 1.1. The first non-comment line of
a file is `mode systems`; the default with no declaration is numeric mode and is
unchanged. The lexer must produce it as an ordinary keyword-plus-identifier
pair, the parser must accept it only as the first statement, and everything
below is refused outside it.

*Go bootstrap:* no concept of a mode. `internal/parser/parser.go` `parseStmt`
has no case for it and `internal/checker/checker.go` has a single policy.

## NEEDS-2: `I64` with bitwise operators

**Status:** blocking. `src/lex.tw` uses it for every offset, line and column.

A signed 64-bit integer distinct from float64, with `and or xor shl shr not`,
defined wrapping, and explicit `i64()` / `f64()` conversions only. The exact
semantics of those six, including that `shr` is arithmetic and that shift counts
are masked to 0..63, are specified in `docs/language-guide.md`, Operators →
Bitwise operators on `I64`; see NEEDS-85 for why they are specified there.
`src/lex.tw:131` (`is_utf8_continuation`) and `src/lex.tw:498` (`utf8_width`)
mask lead bytes with `and`, which is the whole reason the subset needs an
integer type rather than a float that happens to hold integers.

*Go bootstrap:* every numeric value is `*tensor.Tensor`. `internal/interp` has
no integer type and no bitwise operator; `builtins.go` `int` truncates a float
and returns a float.

## NEEDS-3: `enum` with payloads and exhaustive `match`

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

## NEEDS-4: generics: `Arr[T]`, `Dict[K,V]`, `Opt[T]`, `Res[T,E]`

**Status:** blocking.

Type parameters on structs, enums and functions. No bounds, no traits,
monomorphized. Used on nearly every line of `src/`; `src/lex.tw:198` (`Lexer`
holds `Arr[Token]` and `Arr[Comment]`) is the first.

*Go bootstrap:* Go generics for none of it. `internal/value` has `List` as
`[]Value` (heterogeneous) and `Record` as ordered string keys, and the checker
models them as `tList` and `tRecord` with no element type.

## NEEDS-5: `struct`: nominal, mutable, reference semantics

**Status:** the semantics are now decided and written down; the implementation
is still blocking. `src/lex.tw:198` `Lexer` is a cursor that advances. **The
normative text is `docs/language-guide.md`, `struct`, and what a parameter is.**
It states reference semantics for `struct`, `Arr`, `Dict` and `Bytes` together,
including mutation through a field of another struct, and it says that copying
is always explicit. `docs/design.md`, Two modes, and where they disagree, has
the reason the rule is stated rather than left to the implementation.

Fields typed, mutable in place, passed by handle. `advance(lx)` at
`src/lex.tw:240` mutates `lx.i`, `lx.line` and `lx.col` and the caller sees it.
Threading those three through every scan function and returning updated copies
would roughly double the lexer and make its diffs unreadable.

Must stay a distinct type from `Record`: `grad` over a record depends on records
not aliasing (`docs/self-hosting.md` section 1.2).

*Go bootstrap:* `value.Record` exists but is value-ish and has no declared
mutability. `interp` supports record field assignment only through rebinding.

## NEEDS-6: indexable, sliceable `Str`

**Status:** blocking.

`s[i]` yields the `I64` byte value, `s[a:b]` yields a `Str` copy, `len(s)` is
the byte length. `src/lex.tw` is built on this: `scan_ident` at
`src/lex.tw:378` returns `lx.src[start:lx.i]` rather than accumulating, which
is both faster and the only way the token value is guaranteed byte-identical to
the source span.

*Go bootstrap:* `value.Str` is a Go string with no index, no slice, no length
and no concatenation. A lexer written in twill today cannot read its first
character.

## NEEDS-7: `Bytes`: a growable byte buffer

**Status:** blocking. The whole of `src/bytes.tw`.

`bytes_new`, `bytes_push`, `bytes_to_str`. Everything the compiler prints is
built by appending. `src/bytes.tw:41` (`concat`) exists so that the rest of the
compiler never builds a string by repeated `+`, which is quadratic.

*Go bootstrap:* `strings.Builder`, used exactly this way in
`internal/lexer/lexer.go` and `internal/format/format.go`.

## NEEDS-8: `Dict[Str, V]` with insertion-ordered iteration

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

## NEEDS-9: a byte literal

**Status:** not blocking. Worked around in `src/lex.tw:91`.

There is no character literal, so `src/lex.tw` defines `ch(s) = s[0]` and writes
`let C_QUOTE = ch("\"")`. That is readable and it is a runtime call and a
one-byte allocation per constant at module load. A `b'x'` literal folding to a
constant would remove both. Low priority: the workaround is honest and the cost
is paid once.

*Go bootstrap:* Go rune literals, `'\n'` and friends, throughout the lexer.

## NEEDS-10: `Res[T, E]` and postfix `?`

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

## NEEDS-11: `abort(msg)`

**Status:** blocking. `src/lex.tw:45` and about thirty other places.

A panic that means a compiler bug, never a user error. Every `abort` in `src/`
should be unreachable for any input; that is a review rule and it is worth
keeping.

*Go bootstrap:* Go `panic`, used sparingly; most impossible cases in
`internal/interp` return an error instead, which conflates them with user
errors.

## NEEDS-12: `continue` in `while` and `for`

**Status:** specified; the parser and interpreter change is still blocking.
`src/lex.tw:305` and the whole scanner loop. **The normative text is
`docs/language-guide.md`, Control flow → `break` and `continue`**, which covers
`break` as well: innermost loop only, no labels, statements rather than
expressions, no crossing a function boundary, and keywords in systems mode only
so that a numeric-mode `let break = 3` keeps working.

The Go lexer's main loop is a chain of `continue`s and the twill port is the
same shape. Rewriting it as nested `else` would nest eight deep.

*Go bootstrap:* Go `continue`. `internal/interp/interp.go` implements `for` and
`while` with no loop-control statements at all; the language has neither
`break` nor `continue`.

## NEEDS-13: `Unit` as a value, and `unit` as its literal

**Status:** not blocking; cosmetic. `src/lex.tw:421`.

`scan_string` returns `Res[Unit, SyntaxError]`, so it needs something to put in
the `Ok`. The checker already has a `tUnit` type; the value side is missing.

*Go bootstrap:* `value.Unit`, which exists (`internal/value/value.go`) but has
no source syntax.

## NEEDS-14: a `Bool` type name in annotations

**Status:** specified; the parser change is still blocking, trivially. **The
normative text is `docs/language-guide.md`, Systems-mode types → `Bool`.**
`Bool` is a type name in systems mode, legal anywhere a type is, with no
conversion to or from `I64` in either direction.

`src/lex.tw:61` annotates a struct field `trailing: Bool`. The parser currently
reads a bare name after `:` as a record type or a unit
(`internal/parser/parser.go` `parseParam`), so `Bool` would be resolved as a
unit and reported as undeclared.

*Go bootstrap:* `checker.tBool` exists; there is no way to write it.

## NEEDS-15: lexer divergence: non-ASCII whitespace in comments

**Status:** known, accepted divergence, **confirmed by differential test**.
`src/lex.tw:466` (`is_space`).

Go's `strings.TrimSpace` trims a Unicode space set that includes U+0085 and
U+00A0. `src/lex.tw` trims the ASCII members only, because matching the rest
needs a UTF-8 decoder in the scanner for a case that cannot occur in a comment
without the file already being unusual. If the differential harness ever trips
on it, the fix is a decoder in `trim_space`, not in the scanner.

---

## Parser

## NEEDS-16: recursive enum payloads without explicit indirection

**Status:** blocking. `src/ast.tw` throughout.

`Expr` holds `Expr`. The subset's answer is that an enum payload is a struct and
a struct is a handle, so the recursion needs no `Box`. That has to actually be
true in the implementation, including for a payload that is the enum itself
(`Unary { operand: Expr }`).

*Go bootstrap:* interfaces are already references.

## NEEDS-17: a growable `Arr` with `pop` and `set`

**Status:** blocking. `src/parse.tw` builds every statement list this way.

`append(xs, x)` returns a new list today, so building an n-element list is
quadratic. `docs/self-hosting.md` section 1.2 makes the same argument for the
token stream.

*Go bootstrap:* Go slices with `append`.

## NEEDS-18: `f64_of_str` / `str_to_f64`

**Status:** blocking. `src/parse.tw` `parse_number`.

The parser turns a NUMBER token's text into a float and must produce bit-exact
agreement with Go's `strconv.ParseFloat`, or two implementations disagree on
`0.1` and every downstream canonical dump differs. Correct decimal-to-binary
rounding is not something to reimplement in twill: this must be a runtime
primitive that calls the same conversion the Go side does.

*Go bootstrap:* `strconv.ParseFloat(t.Value, 64)` in
`internal/parser/parser.go` `parsePrimary`.

## NEEDS-19: `i64_of_str`

**Status:** blocking. `src/parse.tw` shape dimensions and unit exponents.

The integer equivalent, matching `strconv.Atoi` including its overflow
behaviour, since `parseDim` reports "shape dimension must be a non-negative
integer" on a failure and the two implementations must fail on the same inputs.

*Go bootstrap:* `strconv.Atoi`.

## NEEDS-20: string formatting

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

## NEEDS-21: identity of a heap value (`is_same`)

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

## NEEDS-22: `Opt[T]` returned from `Dict` lookup, and `match` on it

**Status:** blocking. Every environment lookup in `src/check.tw`.

Go's `v, ok := m[k]` is two returns; twill has one. `Opt` is the whole reason
`Res`/`Opt` are in the subset.

*Go bootstrap:* the comma-ok form.

## NEEDS-23: sorting a `Arr[Str]`

**Status:** the ordering is specified; a `sort` builtin is still open.
`src/check.tw` `unit_string`. **The normative text is
`docs/language-guide.md`, Strings → Ordering**, which pins bytewise-unsigned
lexicographic order, shorter-is-smaller on a shared prefix, so that the eleven
hand-written sorts in the ecosystem agree with `sort.Strings` and with each
other. It also records that `<` stays undefined on `Str`, which is why
`str_greater` is a function.

`internal/checker/checker.go` `unitString` calls `sort.Strings` on the unit's
base names before joining, so that `USD*year^-1` renders the same regardless of
map order. The twill version must sort identically (bytewise ascending) or unit
diagnostics differ between implementations.

`src/check.tw` implements an insertion sort rather than asking for a builtin:
the lists are two or three elements, and a `sort` builtin over `Arr[Str]` is a
comparator question that the subset does not need to answer yet.

*Go bootstrap:* `sort.Strings`.

## NEEDS-24: integer division and modulo on `I64`

**Status:** specified; the implementation is still blocking. `src/check.tw`
`unit_sqrt` (`v % 2`, `v / 2`). **The normative text is
`docs/language-guide.md`, Operators → Integer division and modulo on `I64`**,
and NEEDS-44 is the same entry filed twice. `/` truncates toward zero, `%` takes
the sign of the dividend, `MIN_I64 / -1` wraps to `MIN_I64`, and a zero divisor
aborts with a diagnostic rather than returning a `Res`.

Float division would give `1.5` where the checker needs a failure. Defined
behaviour on division by zero as an error value, per
`docs/self-hosting.md` section 1.2.

*Go bootstrap:* Go's `/` and `%` on `int`.

---

## Evaluator and tensors

## NEEDS-25: a foreign call into the native tensor core

**Status: superseded by NEEDS-68. Not blocking; the thing it asks to call does
not exist.** This entry asks for a calling convention into "the native tensor
core", which meant the Go bootstrap's `internal/tensor`. Under the no-Go rule
there is no such core to call: NEEDS-68 opens by stating that there is no
bootstrap to call into, which is why the transcendental primitives have to be
native rather than foreign. A foreign-call convention into a runtime that will
not exist is not a language feature that is missing, it is a question that was
overtaken.

What survives is a real question with a different shape: which primitives the
eventual runtime must provide, and how the checker types them. NEEDS-68 asks
exactly that for the transcendentals, and is the entry to extend. The text below
is kept for section 2.2's line between what reads twill source and what executes
it, which still holds.

*(Original text follows.)*

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

## NEEDS-26: closures capturing a mutable environment

**Status:** blocking. `src/eval.tw` function values.

`internal/interp` closes over a `*Env` with a parent pointer and assignment
walks up to the defining scope. Twill closures exist; what is unspecified is
whether a captured variable is captured by handle or by value, and the
evaluator needs by handle to reproduce the bootstrap's behaviour for
`for i in ... { fns = append(fns, fn() = i) }`.

*Go bootstrap:* `interp.Env` with `assign` walking parents
(`internal/interp/interp.go`).

## NEEDS-27: deep equality with the "different types are never equal" rule

**Status:** blocking. `src/eval.tw` `==`.

The bootstrap's rule, tested in `internal/interp/equality_test.go`: values of
different types compare unequal rather than raising. The twill version must
match it including the new subset types (I64 to I64 by bits, never equal to a
tensor).

*Go bootstrap:* `interp.valuesEqual`.

## NEEDS-28: `read_file`, `write_file`, `args`, `exit`, `write_out`

**Status:** blocking. `src/main.tw`.

`read_file(path) -> Res[Bytes, Str]` and the rest of the process interface from
`docs/self-hosting.md` section 1.2. `src/main.tw` is a CLI and cannot exist
without them.

This is the only entry in this file that widens what an arbitrary `.tw` file can
do, and it should be landed knowing that.

*Go bootstrap:* `os.ReadFile` and friends in `cmd/twill/`.

## NEEDS-29: a stable canonical rendering of a float

**Status: superseded by NEEDS-87. Read that entry first.** The text below is
kept because it is the record of what was believed, but its advice is now wrong
in two ways: there is no bootstrap left to call into, so "treat it as a runtime
primitive" is not an available option; and there is not one canonical rendering
but three, with different verbs and different callers, which NEEDS-87 tabulates.
`src/float.tw` answers this entry. Anyone acting on the paragraph beginning
"Treat it as a runtime primitive" will implement the wrong function.

*(Original text follows.)*

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

## NEEDS-30: recursion depth, and a guard on it

**Status:** not blocking; an operational risk.

The parser and the checker are recursive descent over user input and the
bootstrap has the same exposure. The Go side survives on the goroutine stack;
what the twill side does on a 10,000-deep nesting is undefined until the VM
exists. Worth a depth counter with a diagnostic rather than whatever the VM
does by default.

*Go bootstrap:* none. A deeply nested twill file crashes the Go parser today.

## NEEDS-31: deliberate divergence: `t[]`

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

---

## Found by differential testing of the lexer

The three entries below came out of running `src/lex.tw` against
`internal/lexer/lexer.go` over 385 corpus files and 4,000 fuzzer cases. See
"Verification" at the end of this file.

## NEEDS-32: Go-compatible `%q` for a non-printable or non-ASCII character

**Status:** open divergence. `src/lex.tw:478` (`quote_char`).

The message `unexpected character %q` is the lexer's only use of `%q` on source
text. Go renders a non-printable rune as an escape: a NUL byte prints as
`"\x00"` and a lone surrogate as `"\ufffd"`. `quote_char` emits the raw byte
between quotes, so the two implementations produce different bytes for a NUL,
a vertical tab, or any other non-printable input character.

Every printable case agrees, including multi-byte ones: `€` and an emoji both
round-trip. So this only fires on input that is already malformed, which is
exactly why it would survive a weak harness and needs to be written down.

Fix with the `%q` rendering asked for in NEEDS-20, not with a special case here.

*Go bootstrap:* `fmt.Sprintf("unexpected character %q", string(ch))`.

## NEEDS-33: the Go bootstrap panics on a trailing backslash at end of file

**Status:** a bug in `internal/lexer/lexer.go`, not in `src/lex.tw`.

Source ending in an unterminated string whose last byte is a backslash, for
example `x = "ab\`, makes the Go lexer index past the end of its rune slice and
panic. The string branch consumes the backslash and calls `advance()` for the
escaped character without checking that one exists.

`src/lex.tw:405` checks, and returns "unterminated string" at the opening quote,
which is the right diagnosis: the file's problem is the missing close quote, not
the backslash.

This is a divergence the harness will report, and the resolution is to fix the
Go lexer rather than to reproduce a panic. It is listed here because it is the
first thing self-hosting found, and finding it is the argument
`docs/self-hosting.md` section 3 makes for the exercise.

---

## Verification

`src/lex.tw` was checked against `internal/lexer/lexer.go` by transcribing both
into a single executable form and comparing them, rather than by reading them
side by side. The comparison covers token kind, literal text, line and column,
the comment list including each comment's trailing flag, and the error message
and position on inputs that fail.

- **385 files**: every `.tw` file in `examples/`, `std/`, `testdata/` and
  `src/`. Zero divergences.
- **4,000 fuzzer cases**, seeded, mixing random token soup with mutated slices
  of the corpus, over an alphabet that includes non-ASCII text, escape
  sequences, NUL, vertical tab, unterminated strings and every multi-character
  operator. 2,516 of the cases were error cases. Zero divergences.
- **22 targeted edge cases**. Three divergences, all of them the ones recorded
  above: NEEDS-15, NEEDS-32 and NEEDS-33.

The byte-versus-rune question the port turns on is settled by this rather than
by argument: the column counter in `src/lex.tw` skips UTF-8 continuation bytes,
and the case "multibyte then column-sensitive tokens" confirms that a token
following a multi-byte string literal lands on the same column in both
implementations.

This is not the real harness. The real one runs `src/lex.tw` on a twill runtime
and compares against the Go binary, and it cannot exist until the entries above
are implemented. Until then this is the strongest available evidence, and it is
strong enough to have found NEEDS-33.

---

# What the command line needs from the language

Appended by the CLI work. `src/term/` and `src/cli/` are the twill command line
written in twill, and they rest on everything above plus the entries here. Same
conventions: what the feature is, which file reaches for it, what the Go side
does in the same place.

## NEEDS-34 - `chr(n)` for a single byte

**Status:** blocking. Nothing in `src/term/` emits an escape sequence without it.

Twill string literals recognise `\n`, `\t`, `\"` and `\\` and nothing else
(`docs/language-guide.md`), so ESC (27) and BEL (7) cannot be written. Needed:
`chr(n: I64) -> Str` producing the one-byte string for `n` in 0..255, and it
must be a byte and not a codepoint, because `src/cli/banner.tw` hand-encodes
U+2800 braille as three bytes and would be encoding an encoding otherwise.

*Reaches for it:* `src/term/ansi.tw` (`esc`, `bel`), `src/cli/banner.tw`
(`braille`).

*Go bootstrap:* `internal/builtins` has no `chr`. The Go side writes the escape
introducer as a string literal.

*Alternative that would also do:* an `\x1b` escape in the lexer's string
scanner. `chr` is preferred because the braille packing needs arithmetic on the
byte anyway.

## NEEDS-35 - `Str` concatenation with `+`

**Status:** specified; the implementation is still blocking. Every renderer
builds its output by concatenation. **The normative text is
`docs/language-guide.md`, Strings → Concatenation.** `Str + Str` exists and
produces a new `Str`; `+` between a `Str` and a non-`Str` is an error with no
coercion; and the quadratic cost of building in a loop is stated there along
with the `src/bytes.tw` builder that is the answer to it.

`docs/self-hosting.md` gives `Bytes` a `concat` and gives `Str` length, byte
indexing and slicing, but never says `Str + Str`. The CLI is almost entirely
string building, and doing it through `Bytes` would mean a conversion at every
one of several hundred sites.

*Reaches for it:* every file in `src/term/` and `src/cli/`.

*Go bootstrap:* `+` on `value.Str` is currently an error; the interpreter's
binary op dispatches only tensors.

*Note on cost:* naive concatenation in a loop is quadratic. `src/term/width.tw`
`repeat` and `src/cli/progress.tw` `bar` both build strings a cell at a time, so
either the implementation ropes them or a `Bytes` builder is exposed and these
files are rewritten against it. Flagging it now rather than discovering it on a
200-column progress bar.

*Already half answered.* `src/bytes.tw` exists and wraps exactly this surface
(`bytes_new`, `bytes_push`, `bytes_to_str`, plus `concat`, `join` and `repeat`).
The CLI should be moved onto it once the primitives land, which would delete the
private `join` in `src/term/ansi.tw` and the private `repeat` in
`src/term/width.tw`. They are written out here only because `src/term/` was
built before that file existed.

## NEEDS-36 - `arr(...)` as a literal constructor

**Status:** blocking.

`docs/self-hosting.md` specifies `Arr[T]` with `push`, `pop`, index and `slice`,
but no way to write one down. `arr()` for empty and `arr(a, b, c)` for a
populated one, with `T` unified from the arguments.

*Reaches for it:* `src/cli/help.tw` `groups()` builds the entire help screen as
nested `arr(...)` literals; `src/cli/spinner.tw` `glyphs`.

*Go bootstrap:* `list(...)` exists and is the model to copy, but returns the
heterogeneous `value.List`.

## NEEDS-37 - `Opt[T]` and `match`, for `env`

**Status:** blocking for `src/term/caps.tw`.

`env(name) -> Opt[Str]` from `docs/self-hosting.md` section 1.2 needs the enum
and the `match` that reads it. `caps.tw` `env_or` and `has_env` are the only
uses in the CLI, but they are on the path of every command, so the whole
capability layer is behind this.

*Go bootstrap:* `os.Getenv` returning a value and a found flag.

## NEEDS-38 - `is_tty_stdout()` and `window_size()`

**Status:** blocking for anything that decides to animate.

Two runtime queries not in `docs/self-hosting.md` section 1.2, which lists only
`read_file`, `write_file`, `stdin_all`, `write_out`, `write_err`, `args`, `env`
and `exit`.

- `is_tty_stdout() -> Bool`. Whether stdout is a character device. This is the
  single most important call in the CLI: it is what keeps escape sequences out
  of a redirected log, and there is no way to infer it from the environment.
  Note that stderr needs the same question asked separately, since diagnostics
  go there while a progress bar goes to stdout, and one may be a pipe while the
  other is not.
- `window_size()` returning columns and rows, from `TIOCGWINSZ` on unix and
  `GetConsoleScreenBufferInfo` on Windows, with a zero result meaning unknown
  rather than an error.

*Reaches for it:* `src/term/caps.tw` `detect`.

*Go bootstrap:* neither exists. The current CLI writes to stdout
unconditionally.

*Not asked for:* SIGWINCH. A resize mid-frame smears one repaint and corrects
itself on the next, which is an acceptable cost for not needing a signal
interface in the language.

## NEEDS-39 - a monotonic millisecond clock

**Status:** blocking for the spinner and the progress bar.

`now_ms() -> I64`, monotonic, unaffected by wall-clock adjustment. Every
animated thing in `src/cli/` is driven by it: the frame rate limit in
`src/term/frame.tw`, the spinner's delay gate, the progress bar's smoothed rate
and its estimate.

Monotonic specifically, not wall clock. An NTP step backwards during a long
training run makes a wall-clock rate negative and the estimate nonsense, and
that is a bug that only appears on long runs, which are exactly the runs where
the estimate matters.

*Threaded, not called.* No file in `src/cli/` calls it: the current time is a
parameter to every function that needs it. That is deliberate, so the renderers
stay pure and can be tested by feeding them a clock, and it means this entry is
only needed by whatever drives the loop.

*Go bootstrap:* the standard library clock, not used for any of this yet.

## NEEDS-40 - `F64` in systems mode, with `cos` and the conversions

**Status:** the type question is answered; `cos` and `sin` on it are still
blocking for `src/cli/banner.tw` and `src/cli/tensor.tw`. **The normative text
is `docs/language-guide.md`, Systems-mode types → `F64`, and what a
systems-mode scalar is.**

The answer to the second half of this entry, which is the half that was worth
asking: a systems-mode scalar is **not** a rank-0 tensor. `F64` is a machine
word with no shape, no tape entry and no allocation, so loom's `Meter.total`
does not allocate per step. The float math builtins are entry 15 of
`docs/roadmap.md` and are still open; what is now fixed is that they will take
and return `F64` rather than a rank-0 tensor.

`docs/self-hosting.md` says systems mode has no tensors, and is right, but it
does not say what a plain float is in systems mode. Two files need one:

- `src/cli/banner.tw` computes the ribbon from `cos`, because the mark is
  generated from the twist rather than pasted in as glyphs.
- `src/cli/tensor.tw` formats tensor elements and cannot do that in integers.

Needed: `F64` as a systems-mode scalar type distinct from a rank-0 tensor, the
`f64()` and `i64()` conversions already specified in section 1.2, and `cos` and
`sin` usable on it.

*Go bootstrap:* these are tensor builtins; in numeric mode a scalar is a rank-0
tensor and the distinction does not arise.

## NEEDS-41 - a read-only tensor view for the formatter

**Status:** blocking for `src/cli/tensor.tw` and the REPL.

The REPL's job is printing tensors, and systems mode has none. `tensor.tw`
declares a `View` of dimensions plus row-major elements and formats that, so
what is needed is one bridge: a builtin that turns a numeric-mode tensor into
those two arrays.

`view_of(t)`, or a pair of `shape_of` and `elements_of`. Read-only and by copy.
The copy is the point: the formatter must not be able to alias a live tensor,
and a 512x512 matrix is elided down to nine rows anyway, so a lazy view would be
an optimisation of the wrong thing.

*Go bootstrap:* the tensor type already carries its dimensions and a flat data
slice, so this is an exposure rather than an implementation.

## NEEDS-42 - struct field mutation through a handle

**Status:** specified; the implementation is still blocking for
`src/term/frame.tw`, `src/cli/spinner.tw`, `src/cli/progress.tw`,
`src/cli/repl.tw`. **The normative text is `docs/language-guide.md`, `struct`,
and what a parameter is**, and it covers the case this entry adds: mutation is
visible through a field of another struct, to any depth.

Reference semantics for structs, as specified in `docs/self-hosting.md` section
1.2. Every stateful widget is a struct whose fields a function mutates and whose
caller sees the change: `frame.paint` updates `height`, `last` and `next_due`;
`progress.advance` updates `done` and the smoothed rate; `repl.feed` updates the
bracket depth.

This is already in the design, but it is worth recording that the CLI is its
second consumer after the lexer, and that the CLI needs the mutation to be
visible through a field of another struct (a `Spinner` holds a `Frame`, and
`step` mutates through it), which is a case a lexer never exercises.

## NEEDS-43 - `Arr[T]` element assignment

**Status:** blocking for `src/term/width.tw` `indent_after_first`.

`docs/self-hosting.md` gives `Arr[T]` an index and a `push`, and its feature
list says "indexed assignment" while the summary table does not repeat it.
Recording the use so it is not dropped: `width.tw` rewrites wrapped lines in
place to add the continuation indent.

## NEEDS-44 - integer division and modulo on `I64`

**Status:** the semantics question is answered; the implementation is still
blocking. **The normative text is `docs/language-guide.md`, Operators → Integer
division and modulo on `I64`.** It records the answer this entry asked for,
truncation toward zero with `%` taking the sign of the dividend, and adds the
two things this entry did not ask about and that a caller needs: numeric mode's
`%` is floored and therefore disagrees, and `shr` is `floor(a / 2^k)` and
therefore also disagrees for a negative dividend. See NEEDS-24, which is this
entry under an earlier number.

`/` and `%` exist, but on tensors they are float operations. On `I64` they must
truncate toward zero, and `%` must take the sign of the dividend. The 256-colour
quantisation in `src/term/color.tw`, the braille packing in
`src/cli/banner.tw`, and the eighth-block arithmetic in `src/cli/progress.tw`
are all exact integer arithmetic and are wrong under any other rounding.

Also needed: division by zero on `I64` as an error value rather than a panic,
which section 1.2 already specifies.

## NEEDS-45 - `str()` on `I64`

**Status:** specified; the implementation is still blocking. **The normative
text is `docs/language-guide.md`, Standard library → `str` on a number.**

`str(n)` for an `I64` must produce the digits with no decimal point and no
exponent. Today `str` on a scalar goes through the tensor printer, and a
trailing `.0` would land in every line number, every column count and every axis
index in every diagnostic.

*Measured, because the entry reads as though it had been:* the Go bootstrap
does not emit the trailing `.0`. `internal/value.FormatNumber` returns
`strconv.FormatInt` for any float that is integral, so `str(3)`, `str(scalar(3))`,
`str(sum([1.0, 2.0]))` and `str(len(range(3)))` all print `3`, and `twill fmt`
rewrites the literal `3.0` to `3`. The hazard is real but it is prospective: it
is what a systems-mode `str` would do if it were routed through the tensor
printer, which is what the entry is asking not to happen. The guide also pins
the `F64` rendering and the boundary between the two, since `src/fmt.tw`
`format_number` sends an integral `F64` to `str(k)` on an `I64` and the two
renderings have to agree there.

*Reaches for it:* everywhere. `src/cli/diagnostic.tw` alone uses it for the
line, the column and the gutter width.

## NEEDS-46 - `Str` equality must survive the `Str` rewrite

**Status:** specified, and still a constraint rather than a feature. **The
normative text is `docs/language-guide.md`, Strings → Equality**, which states
it as bytes with no case folding, no normalization and no locale, and Strings →
Ordering, which says that `<` and friends stay undefined on `Str` and pins the
bytewise order the hand-written comparisons implement.

`==` and `!=` on `Str` already work by the deep-equality rule in
`docs/language-guide.md`, and `src/term/caps.tw` leans on it for every
environment comparison. `docs/self-hosting.md` flags making `Str` a distinct
indexable value as the medium-risk change in the whole subset, so this entry
exists to say that the CLI is a second consumer of that rule holding.

## NEEDS-47 - a line reader for the REPL

**Status:** not blocking. `src/cli/repl.tw` is written around not having it.

`repl.tw` owns the prompt and the framing and does not own the read loop,
because line editing, history and completion are a terminal-raw-mode problem
that does not belong in the language. Recorded so the seam is deliberate: the
host reads a line and hands it to `repl.feed`, and the only thing the language
needs is `stdin_all` or a `read_line`.

The one thing the host must get right is restoring the terminal on exit, which
`src/term/frame.tw` `abandon` covers for the frame path but which the line
reader must do for its own raw mode.

## NEEDS-48 - `write_out` and `write_err` taking a `Str`

**Status:** blocking for `src/cli/main.tw`.

Section 1.2 specifies `write_out(Bytes)`. The CLI produces `Str` everywhere and
would otherwise convert at every call site. Either overload accepts a `Str`, or
`Str` to `Bytes` is a zero-copy conversion and that is stated, because if it
copies then the progress bar allocates its whole rendered frame thirty times a
second.

---

## Runtime primitives the compiler names

These are not language features so much as the surface `src/` calls. Listed
separately because they are cheap individually and easy to lose track of.

| Primitive | Used by | Go equivalent |
|---|---|---|
| `arr_new`, `push`, indexed get and set, `len` | everywhere | Go slices |
| `dict_new`, `dict_set`, `dict_get`, `dict_has`, `dict_del`, `dict_keys`, `len` | check.tw, lex.tw, tensor.tw | Go maps, plus `Record.Keys` for order |
| `dict_or(d, k, dflt)` | check.tw unit algebra | the zero value of a Go map read |
| `dict_must(d, k)` | check.tw, eval.tw | a Go map read with a known-present key |
| `bytes_new`, `bytes_push`, `bytes_to_str` | bytes.tw | `strings.Builder` |
| `str(x)` for `I64` and `F64` | every diagnostic | `strconv`, `fmt` |
| `str_quote(s)` | check.tw, parse.tw | `%q` |
| `f64_of_str`, `i64_of_str` | parse.tw | `strconv.ParseFloat`, `strconv.Atoi` |
| `i64_of_f64`, `f64_of_i64` | check.tw, eval.tw | Go conversions |
| `f64_mod`, `f64_pow` | eval.tw | `math.Mod`, `math.Pow` |
| `and(a, b)` and the other bitwise ops | lex.tw | Go `&` |
| `read_file`, `args`, `write_out`, `write_err` | main.tw | `os` |
| `abort(msg)` | everywhere | `panic` |

## NEEDS-49: the systems-mode checker policy

**Status:** open, and it is the design decision, not a coding task.

`docs/self-hosting.md` section 1.3: in systems mode a `TUnknown` surviving to
the end of inference is a diagnostic rather than silence. `src/check.tw`
implements the numeric policy, because that is what `testdata/` was generated
against and what the diagnostics are compared to. The systems policy is a second
policy over the same lattice and it is what would let the checker check `src/`
itself.

Until it exists, the self-hosted compiler is not type-checked by its own
checker, which is an uncomfortable place to be and worth naming.

## NEEDS-50: an out-of-range axis in `transpose`

**Status:** open, low priority. `src/check.tw` `transpose_result`.

An axis outside the tensor's rank makes the result unknown rather than a
diagnostic, matching `internal/checker/checker.go`. Every other axis-taking
builtin reports it through `report_axis`. Fixing this means changing the Go side
too, or the diagnostics diverge.

## NEEDS-51: the import resolver

**Status:** blocking for `src/eval.tw`. `exec_import`.

The policy is written out in the comment there: relative to the importing file,
`std/` resolves to the embedded library unless `TWILL_STD` overrides, `.ra` is
refused outright, a module is evaluated once and cached, and an aliased import
lands in a record while an unaliased one lands unqualified.

What is missing is file reading (NEEDS-28) and a way to reach the embedded
standard library from twill, which the bootstrap does with `go:embed`.

## NEEDS-52: one builtin table, not two

**Status:** open, tidiness. `src/check.tw` and `src/eval.tw`.

The checker's table is what the diagnostics are compared against; the
evaluator's would be what the dispatch uses. They are separate today because
merging them before the dispatch exists would be guessing at its shape.

## NEEDS-53: the formatter

**Status:** done. `src/fmt.tw`, wired into `src/main.tw` `format_file`.

`internal/format/format.go` is ported. What is left of this entry is the
primitives the port names, which are NEEDS-76 and NEEDS-79, and the two
behaviours it inherited rather than chose, which are NEEDS-77 and NEEDS-78.

The rule that had to survive the port, because it is the one people notice:
`fmt --write` never renames a file. The retired-extension check runs before the
file is opened, so such a file is refused and left exactly as it was.

### Verification of the source formatter

`src/fmt.tw` was checked against `internal/format/format.go` by transcribing the
twill formatter, and the lexer and parser it stands on, into one executable
form, and comparing its output against the reference binary's `twill fmt`.

- **405 files** (every `.tw` file in `testdata/`, `examples/`, `std/` and
  `src/`). Zero divergences, zero idempotence failures. 23 of the 405 are
  negative fixtures that both implementations reject, with the same error kind.
- **13,000 generated programs**, over an alphabet built to hit the parts the
  corpus does not: mixed-precedence operator chains, `^` against `-`, unary
  operators over binary operands, slices with either side absent, unit
  expressions with negative and multi-digit exponents, `unit` declarations,
  lambdas with and without block bodies, own-line and trailing comments, empty
  comments, and float literals spanning the exponent thresholds. 8,230 formatted
  on both sides, byte identical, all idempotent. The remaining 4,770 were
  rejected by both, and the rejections agreed on kind: a syntax error on one
  side was a syntax error on the other, a comment the formatter cannot place was
  a refusal on both.

The one divergence the run actually found was the float exponent threshold, and
it is now NEEDS-76. It would not have been caught by the corpus, which contains
no float literal that reaches `%g`.

### Verification of the float conversions

`src/float.tw` was checked by transcribing it into one executable form and
comparing that against two independent references: a correctly rounded
conversion oracle, and the reference binary itself. The second is the one with
teeth, because it compares against the code the differential harness will
actually be diffing against, with no third party's notion of correctness in
between. Three of the reference binary's own commands expose the three
renderings, which is what made it possible:

- **`twill run`, 42,938 printed values.** A generated program of `print(x)`
  lines, compared line for line against `format_number`. Zero divergences. The
  same 42,938 literals parsed back to the same bits, zero divergences.
- **`twill run --dump=canonical`, 30,014 values.** The canonical dump's `num`
  fields against `f64_hex`. Zero divergences. Reading the dump's own hex text
  back gave the original bits in all 30,014, zero divergences.
- **`twill fmt`, 24,621 literals.** The formatted source against `f64_shortest`,
  of which 24,178 reached `%g` rather than `internal/format`'s integer fast
  path. Zero divergences.
- **116,980 cases against the oracle**, covering what the binary cannot easily
  be driven over: 40,000 random f64 bit patterns, 16,000 subnormals of both
  signs, 17,997 values on the `%.6f` half-way boundary, 12,128 integral values
  across the `int64` fast path and its edges, 3,100 values either side of the
  `%g` exponent switch, 3,619 float literals lifted out of `testdata/`,
  `examples/` and `std/`, and 24,074 parse cases including hexadecimal input,
  underscores, and the range and syntax errors. Zero divergences.

287,505 comparisons, zero divergences.

Three things that had to be right and are worth naming because each was a real
divergence at some point in the run rather than a hypothetical:

- **`print` is not `%g`.** See NEEDS-87. Reading `internal/` rather than
  assuming is what turned this up, and assuming would have made every printed
  non-integer wrong.
- **`%g`'s exponent threshold is 6.** The same finding the source formatter's
  run made, from the other side. The corpus cannot catch it: no float literal in
  the tree reaches `%g`'s exponent form at all, so a corpus-only check passes
  while being wrong. It was caught by generated values on both sides of the
  switch.
- **Underscores and hexadecimal on the parse side.** `f64_of_str` first accepted
  `_1.0`, which Go rejects, and rejected `0x1p3`, which Go accepts, because the
  slow path it is ported from relies on a validating reader that runs before it.
  Both were found by generated input, not by the corpus, which contains neither.

The structural differences that could have broken it and did not: the twill
version has no unsigned integer, so the shifts that Go runs on `uint` run
through the helpers in NEEDS-85; and it does the decimal arithmetic on ASCII
digit bytes in an `Arr[I64]` rather than a Go `[800]byte`, which changes every
index expression in the file.

### Verification of the einsum spec parser

`src/tensor.tw`'s `parse_einsum` and `einsum_output_dims` were checked against
`internal/tensor/einsum.go` by the same method: two independent transcriptions,
compared on the parsed spec, the resolved output dimensions, and the exact error
text.

- **12,080 parse cases** (hand-written specs plus 3,000 random ones over an
  alphabet containing `,`, `->`, spaces, uppercase and digits, at operand counts
  0 to 3). Zero divergences.
- **2,728 output-dimension cases**, including unknown sizes, inconsistent label
  sizes, and rank mismatches. Zero divergences.

This matters more than its size suggests, because `src/check.tw` calls the same
parser to validate a literal spec. If the two implementations disagreed, the
checker and the runtime would report different things about the same einsum, and
the corpus would show it as a checker bug rather than a shared one.

The structural differences that could have broken it and did not: the twill
version splits on bytes rather than using `strings.Split`, and it sorts the
implicit output labels with its own insertion sort rather than `sort.Slice`.
Both are places where a port silently drifts.

---

## NEEDS-54: the kernels `src/tensor.tw` does not have yet

**Status:** done. Every row of the table below is written, each with its `vjp_`
gradient rule, and `src/eval.tw` routes to them. The table is kept because it is
the checklist the port was done against.

`src/tensor.tw` implements the kernels in twill and declares its interface at
the head of the file. The builtins in `src/eval.tw` call that interface, and
these are the entries it does not have yet. Each one is named by a builtin that
would otherwise have nothing to call, and each is a kernel plus its gradient
rule, in the style of the ones already there.

| Missing | Wanted by | Go |
|---|---|---|
| `conv2d(a, b) -> Res[Tensor, Str]` | `conv2d` | `internal/tensor/conv.go` `Conv2D` |
| `maxpool2d(t, k) -> Res[Tensor, Str]` | `maxpool2d` | `internal/tensor/conv.go` `MaxPool2D` |
| `diff_axis(t, axis) -> Res[Tensor, Str]` | `diff` | `tensor.DiffAxis` |
| `roll_axis(t, shift, axis) -> Res[Tensor, Str]` | `roll` | `tensor.RollAxis` |
| `cumsum` `cumprod` `cummax` `cummin` `(t) -> Tensor` | the scan builtins with one argument | `internal/tensor/scan.go` |
| `cumsum_axis` `cumprod_axis` `cummax_axis` `cummin_axis` `(t, axis) -> Res[Tensor, Str]` | the same four with an axis | `internal/tensor/scan.go` |
| `from_nested(Nested) -> Res[Tensor, Str]`, `to_nested(t) -> Nested` | `tensor` | `tensor.FromNested`, `(*Tensor).ToNested` |
| `set_record_jets(Bool)`, `hessian(tp, root, leaf) -> Res[Tensor, Str]` | `hessian` | `internal/tensor/jet.go` |
| `new_tape`, `leaf`, `value_of`, `backward`, and the `t_` twins | `grad`, `grads`, `value_and_grad`, `jacobian`, `hessian` | the graph edges on `*Tensor` |

The tape entries are declared in `src/tensor.tw`'s header and not yet written;
they are listed here because `src/eval.tw` calls them today and because their
exact shape, `backward(tp, root, seed) -> Res[Arr[Tensor], Str]` returning leaf
gradients in creation order, is what makes the leaf ordering below `grads`
correct. `grads(loss)(W, b)` returns a list in the order the arguments were
named, and that order is the order `trace_arg` created the leaves in.

Two things about this list are load-bearing and easy to lose.

The `Res[..., Str]` returns are not decoration. `src/eval.tw` `lift` turns a
kernel's message into a runtime error carrying the call's line, and the message
text is compared byte for byte by the differential harness, so a kernel that
aborts instead of returning a message takes the line number with it.

Negative axes are normalised **inside** the kernel and not in `src/eval.tw`.
`src/tensor.tw` `normalize_axis` follows `internal/tensor/ops.go`: it adds the
rank first and then reports the *adjusted* axis if it is still out of range, so
`sum(m, -5)` on a rank-2 tensor says `axis -3 out of range for rank 2`.
Normalising on the eval side would report `-5` and diverge on the single input
that reaches the message.

## NEEDS-55: a seeded random number generator

**Status:** blocking for `src/eval.tw`. `randn`, `rand`, `seed`, `permutation`.

`rng_seed(I64)`, `rng_uniform() -> F64`, `rng_normal() -> F64`,
`rng_perm(n) -> Arr[I64]`. Native: it is one generator's state for the whole
program, which is the thing the language has no way to own.

The contract is reproducibility, and it is stronger than "random": `seed(k)`
followed by the same sequence of `rand`/`randn`/`permutation` calls must give
the same numbers on every run and every platform, because that is what makes a
training run in `examples/` reproducible and what the corpus compares. The Go
side gets this from `math/rand`'s `Float64`, `NormFloat64` and `Perm` on an
explicitly seeded `*rand.Rand`, so the native core has to match those streams
bit for bit, not merely be seeded.

## NEEDS-56: the output sink for `print`

**Status:** blocking for `src/eval.tw`. `bi_print`.

`emit_line(Str)`. Not `write_out`: `interp.New` takes an `out func(string)` and
every caller supplies a different one. The test harness captures into a buffer,
the REPL interleaves with its own prompt, and only the `run` command writes to
stdout. The line ending belongs to the sink, which is why `print` joins with
spaces and adds nothing.

An evaluator that wrote to a file descriptor directly could not be tested by the
differential harness at all, so this is not a detail.

## NEEDS-57: value formatting

**Status: superseded by NEEDS-88. Read that entry first.** This entry asks for
`format_value` and `format_number` together and argues that neither can escape
`src/eval.tw`. Half of it is now false: `format_number(F64)` lives in
`src/float.tw`, which knows nothing about `Value` and never needed eval. Only
`format_value` still has the circular-import problem described below, and
NEEDS-88 states the remaining question accurately. The text is kept for the
reasoning about the import cycle, which is still correct as far as it goes.

*(Original text follows.)*

**Status:** blocking for `src/eval.tw`. `print`, `str`, `write_frame`, and the
`jacobian: f must return a tensor, got %s` message.

`format_value(Value) -> Str` and `format_number(F64) -> Str`, from
`internal/value`'s `Format` and `FormatNumber`. Not ported by anyone: `src/fmt.tw`
is the *source* formatter, the port of `internal/format`, so the obvious name is
taken and this needs a home of its own.

It cannot live in `src/fmt.tw` even if the name were free, because it formats a
`Value`, and `Value` is declared in `src/eval.tw`. A module holding it would
have to import eval, and eval has to call it, so either the two import each
other or it goes in `src/eval.tw`. There is no third option and the language has
no answer for the first one today.

It is also a bigger job than it looks: `FormatNumber` is the float rendering of
NEEDS-29, and every printed number in every `testdata/` expectation goes through
it.

## NEEDS-58: paths resolved against the running source file

**Status:** blocking for `src/eval.tw`. `read_csv`, `read_frame`, `write_frame`,
`save`, `load`.

`resolve_path(Str) -> Str`: an absolute path unchanged, a relative one joined to
the directory of the source file currently executing, *not* the process's
working directory. `internal/interp` keeps a `srcStack` for this so that a
script reading `data.csv` next to itself works when run from anywhere.

The import resolver (NEEDS-51) needs the same stack, and they should be one
thing rather than two.

## NEEDS-59: reading and writing whole files

**Status:** blocking for `src/eval.tw`. The frame builtins.

NEEDS-28 has both, as `read_file(path) -> Res[Bytes, Str]` and `write_file`.
The shapes the frame builtins want differ, and the difference is not cosmetic:

- `read_file(path) -> Opt[Str]`. An option, because `read_csv` reports
  `read_csv: cannot read "..."` and discards the underlying OS error entirely,
  so a `Res` carrying a message the caller must then throw away is the wrong
  shape. `Str` rather than `Bytes` because every line of the file is then split
  and parsed as text.
- `write_file(path, Str) -> Res[Unit, Str]`, likewise taking `Str`. This is the
  same widening NEEDS-48 asks for on `write_out`, for the same reason.

Either NEEDS-28's signatures grow these, or the conversions happen at each call
site and `read_csv` allocates a second copy of the whole file.

## NEEDS-60: parsing a float the way Go does

**Status:** blocking for `src/eval.tw`. `read_csv`, `read_frame`.

`f64_parse(Str) -> Opt[F64]`. The runtime primitive table already lists
`f64_of_str` for the lexer, but the lexer only ever hands it text its own scanner
accepted. This one is handed arbitrary CSV fields and has to decide, so it needs
the option return and it needs Go's exact acceptance set: leading sign,
`inf`/`infinity`/`nan` case-insensitively, hex float literals, and underscores
refused. A parser that accepts a superset turns a corrupt column into silent
numbers; one that accepts a subset rejects files the bootstrap reads.

## NEEDS-61: `trim_space` over Unicode, not ASCII

**Status:** open, low priority. `src/eval.tw` `trim_space`.

`strings.TrimSpace` strips every Unicode space, including U+00A0 and the
ideographic space. The twill version strips the six ASCII ones, which is every
byte a CSV realistically contains and not every byte Go would strip. A file with
a non-breaking space around a number parses on the Go side and fails here.

Either a `trim_space` primitive with Go's semantics, or `unicode.IsSpace` as a
predicate the twill loop can call.

## NEEDS-62: `Nested`, and where it belongs

**Status:** done. `Nested` is declared in `src/tensor.tw` next to `from_nested`
and `to_nested`, and `src/eval.tw` `value_to_nested` builds a `tensor.Nested`.
The stopgap copy in `src/eval.tw` is gone.

`tensor([[1, 2], [3, 4]])` goes through an intermediate that is a number or a
list of them nested to any depth, and `tensor.FromNested` reads the shape off it
and refuses a ragged one. `src/eval.tw` declares the enum because
`src/tensor.tw` does not have it, which is the wrong place: `from_nested` and
`to_nested` are the tensor engine's and the type they speak should be too. Moving
it is a one-line change once the kernel set lands, and it is recorded so it does
not become permanent by default.

## NEEDS-63: opaque values from the native core

**Status:** blocking for `src/eval.tw`. `gbm_fit`, `gbm_predict`.

A fitted model is a value twill holds and cannot look inside. `src/eval.tw` adds
`VForeign(ForeignVal { kind, handle })` for it: a string naming the kind and an
opaque handle the core resolves. `gbm_predict` checks the kind, which is what
makes `gbm_predict(3, X)` report *first argument must be a model from gbm_fit*
rather than crashing in the core.

The primitives:

- `gbm_fit(Arr[F64], Arr[F64], I64, I64, GbmParams) -> Res[I64, Str]`
- `gbm_predict(I64, Arr[F64], I64, I64) -> Res[Arr[F64], Str]`

`internal/gbm` is 900 lines of tree building and it is the one part of the
builtin surface that is neither a tensor kernel nor twill's business. It stays
native. The handle has to survive `save` and `load` (NEEDS-64), which is the
part that makes it more than an integer.

What is *not* deferred: `GbmParams` and its defaults are declared in
`src/eval.tw`, because the defaults are part of what `gbm_fit(X, y)` means to
someone reading a twill program.

## NEEDS-64: `save` and `load`

**Status:** blocking for `src/eval.tw`.

`save_value(Value, Str) -> Res[Unit, Str]` and
`load_value(Str) -> Res[Value, Str]`, from `internal/interp/serialize.go`.

The format is the contract: a file written by the bootstrap must load in the
self-hosted evaluator and the other way round, or a model trained before the
switch is lost. That makes this the one primitive here whose *encoding* is part
of the specification rather than an implementation detail, and it covers
tensors, lists, records and the gbm model of NEEDS-63.

Porting the encoder to twill is possible for everything except the model, so the
seam is the same one either way, and it is recorded as native for that reason.

## NEEDS-65: `f64_trunc`, `f64_floor`, `f64_ceil`, `f64_round`

**Status:** blocking for `src/eval.tw`. `int`, `floor`, `ceil`, `round`, and
every `int_of` coercion.

Alongside the `f64_mod` and `f64_pow` already in the runtime primitive table.
`f64_round` is half away from zero, matching `math.Round` and not the
round-half-to-even a reader coming from Python or from IEEE would assume, and
the difference shows on exactly the inputs a test would use.

## NEEDS-66: three builtins the checker's table does not know

**Status:** open, and it is a bug on the checker's side rather than a language
gap. `src/check.tw` `builtin_names`.

`argsort`, `argtopk` and `split` are defined in
`internal/interp/builtins.go` and ported in `src/eval.tw`, and they are absent
from `src/check.tw`'s `builtin_names`. A program calling one of them is reported
as an undefined variable by the checker and then works when run.

The Go checker has the same list, so fixing it means fixing both or the
diagnostics diverge. Related to NEEDS-52, but separate: NEEDS-52 is that there
are two tables, this is that they already disagree.

## NEEDS-67: mutating a struct through a parameter

**Status:** answered. It was a semantics question, not a task, and the answer is
handle semantics, which is what `src/` already assumed. **The normative text is
`docs/language-guide.md`, `struct`, and what a parameter is.** `src/eval.tw`
`gbm_opts_from_record` is correct as written and does not have to return the
params.

That function takes a `GbmParams` and assigns to its fields, and the caller
expects to see the changes. Whether it does depends on whether a struct is
passed by handle or by value, which `docs/self-hosting.md` does not say. The
same question decides whether `Env`, `Tape` and `Printer` work at all, so it is
already answered implicitly everywhere in `src/`, but it is answered by
assumption and not by a rule.

The assumption throughout `src/` is that a struct is a handle and assignment
through it is visible to the caller, exactly as a Go pointer receiver is. If the
answer turns out to be by-value, `gbm_opts_from_record` has to return the params
and several other things in `src/` break more quietly than it does.

---

## Tensor kernels and autodiff

The entries below are what `src/tensor.tw` reaches for now that the kernels and
the gradient rules are in twill rather than deferred to a native core. NEEDS-25
described the calling convention into that core; there is no core, so what it
asked for is replaced by the handful of genuine primitives listed here. Nothing
in `src/tensor.tw` needs a foreign call any more.

## NEEDS-68: the transcendental float primitives

**Status:** blocking for `src/tensor.tw`. `apply_unary`, `d_unary`, `softmax`,
`logsumexp_axis`, `vjp_logsumexp`.

`f64_exp`, `f64_log`, `f64_sin`, `f64_cos`, `f64_sqrt`, `f64_tanh`. These have
to be native primitives, not twill: under the no-Go rule there is no bootstrap
to call into, and a series expansion written in twill would not agree with
`math.Exp` in the last bits.

Agreement in the last bits is the requirement, not a nicety. `testdata/` compares
output byte for byte after a canonical float rendering (NEEDS-29), so an `exp`
that is one ulp off turns every test touching a sigmoid into a diff. Whatever
supplies these must be the same implementation the bootstrap used, which in
practice means Go's `math` or a faithful port of it.

`f64_pow` is already in the runtime primitive table and `f64_floor` is NEEDS-65;
neither is repeated here.

*Go bootstrap:* `math.Exp` and friends, called from `internal/tensor/tensor.go`.

## NEEDS-69: `f64_signbit`

**Status:** open, low priority. `src/tensor.tw` `f64_max`, `f64_min`.

`math.Max(-0, +0)` is `+0`, and a comparison chain cannot tell the two zeros
apart. The only way to reproduce it is to ask which zero it is.

Low priority because the sign of a zero is invisible until something divides by
it, and it is recorded rather than skipped because when it does show up it shows
up as an infinity of the wrong sign in a gradient, which reads as a bug in the
gradient rather than in `max`.

*Go bootstrap:* `math.Signbit`.

## NEEDS-70: equality on a payload-free enum case

**Status:** blocking for `src/tensor.tw`. `is_same_op`, and every dispatch in
`vjp`.

`Op` has forty-odd cases and none carries a payload. Asking whether a value is
`OpAdd` currently means a `match` with forty arms, or `is_same_op`, which
compares the rendered names and is both slow and a lie about what it is doing.

What is wanted is `==` on two values of the same payload-free enum, comparing
the case and nothing else. This is narrower than deriving equality for all
enums, which would have to decide what a payload comparison means, and it covers
the case that actually appears.

Without it `vjp`'s dispatch is a string compare per op per backward pass, which
is not a correctness problem and is an embarrassing one.

*Go bootstrap:* `internal/interp/builtins.go` dispatches on a string name, so it
has the same shape and the same cost, and gets away with it because the Go map
lookup is one hash.

## NEEDS-71: an `Arr` parameter aliases the caller's array

**Status:** answered; the implementation is still blocking for `src/tensor.tw`.
`accumulate`, `odo_step`, `sort_offsets`, and every kernel that fills a buffer
it was handed. **The normative text is `docs/language-guide.md`, `struct`, and
what a parameter is**, which states the `Arr` rule and the `struct` rule in one
place precisely because `Odometer` is mutated through both at once. An `Arr`
parameter aliases; the backward pass does not return zeros.

`accumulate(cot, touched, node, buf)` mutates `cot[node].data` and expects the
caller to see it. `odo_step(odo)` advances a struct's arrays in place. If an
`Arr` parameter is copied rather than aliased, every one of those is a silent
no-op and the whole backward pass returns zeros.

This is the array half of NEEDS-67, which asks the same question about a struct.
The two answers have to agree, because `Odometer` is a struct holding arrays and
is mutated through both at once.

The bootstrap's answer is aliasing, because a Go slice is a header over shared
storage, so that is the answer `src/` is written against.

*Go bootstrap:* Go slices.

## NEEDS-72: nested containers

**Status:** blocking for `src/tensor.tw`. `Arr[Arr[I64]]` in `Odometer.contrib`
and `einsum_plan`, `Arr[Tensor]` in `concat`, `split` and `backward`,
`Arr[Bool]` in `resolve_perm` and `backward`.

`docs/self-hosting.md` section 1.2 lists `Arr[T]` without saying whether `T` may
itself be a container or a struct. Every entry above needs it to be.

Nothing exotic is wanted: no variance, no covariant assignment, just the
element type being any type the subset already has. It is listed because a
straightforward reading of the section is that `Arr` holds scalars, and the
tensor kernels would then need a hand-rolled flattening for each of the five
uses above, which is five chances to get an index wrong.

*Go bootstrap:* `[][]int`, `[]*Tensor`, `[]bool`.

## NEEDS-73: `abort` in value position

**Status:** open, small. `src/tensor.tw` `apply_binary`, `apply_unary`.

Both end in a `_ =>` arm that calls `abort` because the op passed was not of the
kind the function handles. The arm has to have the same type as the others,
which is `F64`, so `abort` has to be usable as an expression of any type and be
understood by the checker as never returning.

The alternative is returning a sentinel float and letting a wrong op silently
compute with it, which is worse in exactly the way this file is trying to avoid.

*Go bootstrap:* `panic` is a statement in Go and these branches are written as
an early `return` there, which is available because the Go functions are not
expression-bodied.

## NEEDS-74: rendering an `Arr[I64]` the way Go's `%v` does

**Status:** open, diagnostics only. `src/tensor.tw` `resolve_perm`.

The invalid-permutation message renders the axes with `shape_string`, which
produces `[1, 0]`. `internal/tensor/ops.go` uses `%v` on a `[]int`, which
produces `[1 0]`, with spaces and no commas.

Every other shape in a diagnostic goes through `shape_string` on both sides and
matches. This one does not, because Go is printing an axis list rather than a
shape. Either the Go side switches to the shape rendering, or a second renderer
exists for it. It is one message and it is written down so the differential
harness's first complaint is not a surprise.

*Go bootstrap:* `fmt.Errorf` with `%v`.

## NEEDS-75: a tape the interpreted code records on

**Status:** done. `src/eval.tw` has `tape_push`, `tape_pop` and `tape_node_of`
over a stack of tapes with dynamic extent, and one `tr_` shim per differentiable
kernel routing to the taped twin while a tape is installed. What is left of this
entry is the two costs it exposed, NEEDS-81 and NEEDS-82.

The original text follows, because the two design decisions in it are the ones
the implementation was made to satisfy and are worth checking it against.

**Was:** blocking for `src/eval.tw`. `grad`, `grads`, `value_and_grad`,
`jacobian`, `hessian`. It is the one hole in `src/eval.tw` that is in the middle
of something rather than at its edge.

`src/tensor.tw` splits every op in two: `binary(op, a, b)` computes, and
`t_binary(tp, op, a, b)` computes and records on a tape, returning a node index.
That split is right, and it is exactly why grad is not just a matter of calling
kernels: the arithmetic that has to be recorded happens inside f, which is
interpreted twill, in code `src/eval.tw` evaluates but does not write.

Three functions stand for the missing piece:

    tape_push(tp)                  make tp the tape ops record on
    tape_pop()                     restore the previous one, or none
    tape_node_of(v) -> Opt[I64]    the node a traced value came from

The Go bootstrap needs none of them, because a `*tensor.Tensor` carries its own
graph edges: a value *is* a node, and `Backward` walks from it. Here a `Tensor`
is a shape and a buffer, node identity lives in the tape, and the association
between the two has to exist somewhere. These three are that somewhere.

Whichever way it is answered, two things follow and both are design decisions
rather than coding:

- **Every builtin needs a taped path.** `relu(x)` inside a loss has to record;
  `relu(x)` in a print statement should not, or every forward pass pins its
  whole history alive. `src/eval.tw` calls the untaped kernel today. The natural
  answer is that each builtin asks whether a tape is installed and picks the
  twin, which doubles nothing but touches every call site.
- **Nesting.** `tape_push`/`tape_pop` are a stack rather than a single slot
  because `jacobian(f)` runs f once per output element and `grad` inside a
  function passed to `map` is legal. A single slot would make the inner call
  silently steal the outer tape's recording.

The alternative, giving `VTensor` a node field, was not taken here because it
would put a gradient concept into the value model that most values never use,
and `src/tensor.tw` deliberately kept `Tensor` a shape and a buffer. Recorded so
the choice is visible rather than assumed.

## NEEDS-76: `f64_shortest`, Go's `%g` for the source formatter

**Status:** blocking for `src/fmt.tw`. `format_number`.

`f64_shortest(F64) -> Str`, which must equal Go's
`strconv.FormatFloat(x, 'g', -1, 64)` exactly. This is not NEEDS-57's
`format_number` (that one renders a `Value` for `print`) and it is not
NEEDS-29's hexadecimal dump form. It is the decimal spelling a number literal
gets when `twill fmt` writes it back out, and it is compared byte for byte
against the Go formatter over the whole corpus.

The contract is narrower than "prints a float", and the differential run caught
the part that is easy to get wrong, so it is written down here rather than left
to the reader of `strconv`:

- Shortest digits that round-trip back to the same `F64`.
- Exponent form when the decimal exponent is `< -4` or `>= 6`. Not 21. Go uses
  a precision of 6 for this decision when the precision is "shortest".
- The exponent is at least two digits and always signed: `1e-07`, `1e+20`.
- Negative zero prints as `-0`.

Measured against the reference binary: `1234567.5` gives `1.2345675e+06`,
`123456.5` gives `123456.5`, `1e20` gives `1e+20`. `1000000` never reaches this
function, because `format_number` takes the integer path first, which is why the
threshold of 6 is invisible until a value has a fractional part.

*Go bootstrap:* `strconv.FormatFloat`, through `internal/format`'s
`formatNumber`.

## NEEDS-77: the formatter drops `unit USD`

**Status:** open, and it is a bug on the Go side first. `src/fmt.tw` `stmt`,
`internal/format/format.go` `(*printer).stmt`.

`internal/format/format.go`'s statement switch has no case for `*ast.UnitDecl`,
so a `unit` declaration formats to nothing and a file that declares a base unit
loses it. `src/fmt.tw` reproduces this deliberately: the cross-agreement check
compares bytes, so fixing it in one implementation alone would turn a silently
dropped declaration into a reported divergence and bury the real problem under
a harness failure.

Fix both together, in one change, or the corpus goes red. The fix is one line on
each side, printing `unit <name>` through the same trailing-comment path every
other single-line statement uses.

## NEEDS-78: the formatter does not preserve blank lines

**Status:** open, cosmetic, and the most likely reason someone stops running
`twill fmt`. `src/fmt.tw`, and `internal/format/format.go` equally.

The printer emits one line per statement and nothing else, so the blank lines a
author put between paragraphs of a function are gone after one format. Comments
survive; the whitespace that groups them does not.

Preserving them needs the tree to carry the gap: the statement line numbers are
already there, so a gap of two or more source lines between consecutive
statements could re-emit as one blank line. That rule is not in the Go file to
copy, which is exactly why it is recorded here rather than invented in the port.
Whatever is chosen has to be chosen on both sides at once, for the same reason
as NEEDS-77.

## NEEDS-79: a `Dict` keyed by `I64`

**Status:** open, a workaround is in place. `src/fmt.tw` `Printer.trailing`,
`line_key`.

The formatter maps a source line to the trailing comment on it. The natural type
is `Dict[I64, Str]`; `docs/self-hosting.md` only specifies `Str` keys, so the
line number is rendered with `str()` at every set and every get. That is a
decimal conversion per statement printed, and it makes the key type a lie about
what the map is for.

Either `Dict` takes any equatable key, or this stays a documented workaround.
It is not blocking and it is not free: the same pattern will appear anywhere the
compiler wants to key on a node id, which `src/ast.tw` hands out precisely so it
can be keyed on.

## NEEDS-80: `twill tokens` is not a command

**Status:** recorded so it is not read as an accident. `src/main.tw`.

The earlier draft of `src/main.tw` had `twill tokens <file>` and
`twill dump --dump=tokens`, and `src/lex.tw`'s `dump_tokens` was written for
them. `cmd/twill/main.go` has neither, and this file is compared against it with
stderr byte for byte, so a command the reference binary does not have would make
`twill tokens x.tw` print a token stream on one side and
`twill: cannot read file "tokens"` on the other.

`dump_tokens` stays in `src/lex.tw` and the lexer's differential check should
call it directly rather than through the CLI. If a token dump is wanted from the
command line, it has to be added to the Go binary first.

## NEEDS-81: a map keyed by value identity

**Status:** open, a workaround is in place and it is quadratic.
`src/eval.tw` `tape_node_of_tensor`.

The tape seam has to answer "which node did this tensor come from", and the
answer has to be by identity: two tensors holding equal data are different graph
nodes, and NEEDS-27's deep equality is the wrong question. The workaround is a
backwards linear scan of `tp.entries` calling `is_same`, so a forward pass over
a tape of n entries costs O(n^2) identity comparisons.

What is wanted is a `Dict` whose key is the identity of a heap value, the thing
NEEDS-21's `is_same` compares, rather than a `Str`. NEEDS-79 asks the adjacent
question for `I64` keys; the two want the same relaxation of the key type from
one concrete type to any type with an equality.

Recorded rather than fixed because the scan is obviously correct and a wrong
answer here does not fail loudly: it returns the wrong node and the gradient
comes back plausible and wrong.

*Go bootstrap:* none needed. A `*tensor.Tensor` is its own node, so the lookup
does not exist there at all, which is why this cost is new rather than ported.

## NEEDS-82: file-level state that is mutated in place

**Status:** half answered, and still blocking for `src/eval.tw`. `TAPES`, and
the three functions around it: `tape_push`, `tape_pop`, `tape_current`.

The aliasing half is settled: **`docs/language-guide.md`, `struct`, and what a
parameter is** says an `Arr` is a handle and copying is always explicit, so
pushing to `TAPES` from anywhere mutates the one array. The other half, whether
a file-level `let` may be initialised by a call and whether that call runs once,
is not answered here and is the same question as weft's entry 9 (`docs/roadmap.md`
entry 28).

`let TAPES: Arr[tensor.Tape] = arr_new()` at file level, pushed and popped for
the dynamic extent of a differentiated call. Two things have to be true for it
and `docs/self-hosting.md` says neither: a file-level `let` may be initialised
by a call rather than only by a literal, and pushing to a file-level `Arr`
mutates the one array rather than a per-reference copy. The existing file-level
bindings in `src/eval.tw` are all `Str` literals, so nothing has tested either.

The alternative, threading the tape through every `eval_*` function and every
builtin, is rejected in the comment above `TAPES` and the reason is semantic
rather than ergonomic: a threaded tape is lexical, and a closure captured before
the `grad` would then carry no tape and record nothing.

*Go bootstrap:* `internal/tensor/jet.go`'s package-level `recordJets`, set and
cleared by `internal/interp/builtins.go` around the graph build. Same lifetime,
same single-threaded assumption; a stack rather than a bool because grad nests
and `SetRecordJets` does not have to.

## NEEDS-83: the ops with no forward-mode rule

**Status:** open, and it is a gap in `hessian` rather than in the language.
`src/tensor.tw` `jet_node`, `is_rearrangement`.

Forward-mode 2-jets are written for the elementwise binary and unary ops,
matmul, sum, mean, cumsum, cumprod, cummax, cummin, where, and the pure
rearrangements: reshape, broadcast_to, transpose, flip, gather, index0, slice0
and split. Everything else reports

    second-order autodiff: the function uses an operation without forward-mode
    support

which is `internal/tensor/jet.go`'s message, byte for byte. The ops that hit it
are sort, topk, concat, prod, median, softmax, logsumexp, conv2d and maxpool2d.

Two different reasons are hiding in that list and they want different work.

Concat, sort and topk are rearrangements whose jet is a permutation of the
tangents, and the permutation is either recorded already or derivable. They are
missing because the shared replay path applies the forward kernel to the tangent
buffer, and doing that to a sort would sort the *tangents*, which is a plausible
wrong answer rather than a loud one. Each needs its own selection-then-apply
rule, in the shape of `jet_select_by_value`.

Softmax, logsumexp, prod, median, conv2d and maxpool2d have no jvp in
`internal/tensor` either, so a `hessian` over them fails on the Go side with the
same message. Adding them here first would make the two implementations disagree
about which programs are second-order differentiable, which the differential
harness would report as a divergence rather than as the improvement it is. Both
sides at once, as with NEEDS-77.

*Go bootstrap:* `internal/tensor/jet.go`, and the `recordJets` guards in
`ops.go`, `scan.go`, `gather.go` and `tensor.go` that show exactly which ops
have a rule there.

## NEEDS-84: `f64_bits` and `f64_from_bits`

**Status:** blocking for `src/float.tw`. Every function in it.

`f64_bits(F64) -> I64` and `f64_from_bits(I64) -> F64`, the IEEE 754 bit
pattern of a double and its inverse. Go's `math.Float64bits` and
`math.Float64frombits`, which are a reinterpretation of the same eight bytes and
not a conversion.

These are the only two primitives `src/float.tw` needs, and that is the point
worth recording. Formatting and parsing a float are entirely integer and string
work once the sign, exponent and mantissa are in hand; what cannot be written in
twill is getting at them. `i64_of_f64` is not a substitute, because it truncates
toward zero and therefore destroys exactly the information wanted.

They are also the only place the systems subset has to admit that `F64` has a
representation. Everywhere else it is a number.

*Go bootstrap:* `math.Float64bits`, reached through `strconv`.

## NEEDS-85: `shr` on `I64` is arithmetic, and there is no unsigned anything

**Status:** the semantics are now decided and written down; the `U64` half is
still open, and a workaround is in place. `src/float.tw` `ushr`, `udiv10`,
`unonzero`. **The normative text is `docs/language-guide.md`, Operators →
Bitwise operators on `I64`.** Read it there, not here.

The decision: `shr` is an **arithmetic** shift, `shl` shifts zeros in, and shift
counts are masked to 0..63. A logical right shift is spelled by building one,
and `src/float.tw`'s `ushr` is the named idiom.

Why it had to be decided rather than measured: the Go bootstrap does not
implement `mode systems`, `I64`, or any bitwise operator, so there was no
running implementation to appeal to. `shr` is a name the systems-mode sources
use and that nothing yet defines. Arithmetic was chosen because it is what the
`internal/strconv` original does on `int64`, because `src/float.tw` and
`std/random.tw` were already written against it and already carry the compensating
helper, and because a language whose only integer type is signed should have its
shift agree with division by a power of two.

The cost of leaving it unstated was not theoretical: loom's `src/rng.tw` had
assumed the opposite, so its splitmix64 finaliser was a different generator, and
nothing would have reported it. That is fixed in loom.

`docs/self-hosting.md` section 1.2 specifies `and or xor shl shr not` on a
two's-complement `I64` and does not say what `shr` does with the sign bit.
`src/float.tw` assumes Go's answer, which is arithmetic: `shr` on a negative
value shifts ones in from the top.

That assumption is load-bearing rather than incidental. The decimal shifts carry
intermediate values up to `10 * 2^60 + 9`, which is inside 64 bits but past
`2^63`, so the sign bit is set on values that are not negative numbers at all.
Go runs that arithmetic on `uint` and picks its shift chunk size for it. The
subset has no unsigned type, so `src/float.tw` carries three helpers:

- `ushr`, a logical right shift, built by clearing the sign bit and putting it
  back at its shifted position;
- `udiv10`, an unsigned divide by ten, built as `(x >>> 1) / 5`, which is exact
  because `floor(floor(n/2)/5)` is `floor(n/10)`;
- `unonzero`, spelled out because `x > 0` is false for precisely the values the
  loops have to keep going on.

Two things would settle this. `shr` being specified as arithmetic, so the
helpers are correct rather than lucky. And, separately, whether the subset wants
a `U64`: the answer is probably no, since three helpers in one file is a smaller
cost than a second integer type in the checker, but the decision should be made
rather than inherited.

*Go bootstrap:* `uint` in `internal/strconv/decimal.go`.

## NEEDS-86: a file-level `let` initialised by a call

**Status:** open, assumed to work. `src/float.tw` `LC_CUTOFF`, `LC_DELTA`,
`POWTAB`.

Three constant tables are built by a function and bound at file level:

    let LC_CUTOFF: Arr[Str] = leftcheat_cutoffs()

`src/lex.tw` already does this for its byte constants, so the form is not new,
but those are scalars from a one-line function and these are 61-element arrays
built with `push`. What is being assumed is that the initialiser runs once, at
module load, before any function that reads it, and that the array it returns is
shared rather than rebuilt per read.

If it is rebuilt per read, `left_shift` reconstructs a 61-entry table on every
call and the formatter's cost goes from linear to quadratic in the digit count,
silently. If the initialisers run in a different order than they are written,
nothing here breaks, because none of the three reads another, but that is luck
rather than design and the next file will not be so lucky.

This is the same question NEEDS-82 asks about mutable file-level state, from the
immutable side.

## NEEDS-87: NEEDS-29 is answered, and its advice is now wrong

**Status:** resolved by `src/float.tw`. Recorded because NEEDS-29 still says the
opposite and a reader will find it first.

NEEDS-29 says a canonical float rendering should be a runtime primitive calling
the same code the Go side calls, and warns that reimplementing Ryu or Grisu in
twill is a way to lose a month. That was correct advice while calling into the
bootstrap was allowed. Under the no-Go rule there is nothing to call, so the
choice is between a port and no float output at all.

`src/float.tw` is the port, and the warning was avoided rather than ignored: it
implements Go's exact multiprecision-decimal path, not Ryu. That path is
definitional rather than heuristic. Go's fast paths are only permitted to answer
when they can prove they agree with it, so porting it reproduces Go by
construction, whereas porting Ryu reproduces Go only if six hundred table
entries survive transcription, and there is no way to run twill to find the one
that did not.

Three renderings came out of reading `internal/`, not one, and this is the part
NEEDS-29 obscures by saying "a stable canonical rendering" in the singular:

| Caller | Verb | Entry point |
|---|---|---|
| `internal/value.FormatNumber`, which is `print` | `'f'`, precision 6, then trailing zeros and a trailing point trimmed, behind an integer fast path | `format_number` |
| `internal/format`, the source formatter | `'g'`, precision -1 | `f64_shortest`, NEEDS-76 |
| `cmd/twill/dump.go`, the canonical dump | `'x'`, precision -1 | `f64_hex` |

Anyone who reads NEEDS-29 and implements `%g` for `print` has written the wrong
function. `print(0.1)` is `0.1` under both, but `print(1/3)` is `0.333333` and
not `0.3333333333333333`, and `print(1e300)` is 301 digits and not `1e+300`.

Two more inherited behaviours, both confirmed against the reference binary
rather than reasoned about, because both look like bugs:

- `print(-0.0)` is `0`. Negative zero equals `float64(int64(-0.0))`, so it takes
  the integer path, which has no sign to print.
- `print(-1e-9)` is `-0`. Precision 6 gives `-0.000000` and trimming leaves the
  sign behind. Any value under half a millionth in magnitude prints as a signed
  zero.

`f64_of_str` is in the same file for the same reason: NEEDS-29's rendering has
to round-trip, and a parse that accumulates digits and multiplies by a power of
ten makes `0.1` a different float from Go's before the program runs.

## NEEDS-88: `format_number` for a `Value` still has no home

**Status:** open, and narrowed. `src/eval.tw`.

NEEDS-57 asks for `format_value(Value) -> Str` and `format_number(F64) -> Str`
together, and argues that neither can live outside `src/eval.tw` because `Value`
is declared there and a module holding them would have to import eval while eval
calls them.

Half of that is now settled: `format_number(F64)` is in `src/float.tw`, which
imports nothing but `bytes.tw` and knows nothing about `Value`. The scalar case
never needed eval.

`format_value` genuinely does, because it walks tensors and records. So the
circular-import problem NEEDS-57 describes is real but smaller than it looked,
and the answer is that `src/eval.tw` imports `src/float.tw` and keeps only the
walk. Recorded rather than left implicit, because the obvious reading of
NEEDS-57 is that both halves have to go in eval, and putting `format_number`
there would mean the source formatter and the checker cannot reach it.

## NEEDS-89: a round-trip float rendering the standard library can call

**Status:** open, blocking for `std/json.tw`. `number_str`.

`f64_shortest(F64) -> Str`, the shortest decimal that parses back to the same
double, reachable from `std/`.

`src/float.tw` already implements exactly this algorithm, for the source
formatter, and answers NEEDS-29 with it. The problem is not the algorithm, it is
where it lives: `src/` is the compiler and `std/` is the library the compiler
compiles, so a std module importing the compiler inverts the dependency, and the
alternative of a second copy in `std/` is the one thing NEEDS-29 warns against
by name.

The reason the obvious substitute does not work is worth stating, because it
looks like it should. `str(x)` is `internal/value.FormatNumber`: `'f'` with a
precision of 6, trailing zeros trimmed, behind an integer fast path. So
`str(1.0 / 3.0)` is `"0.333333"`, and a JSON document rendered through it and
parsed back is not the document it started as. Round-tripping is the one
property a serialiser has to have.

Either `f64_shortest` becomes a runtime primitive alongside `f64_of_str`, or
`src/float.tw` moves somewhere both halves of the tree can reach. The second is
probably right and is a layering decision rather than a coding task.

*Go bootstrap:* `strconv.FormatFloat(x, 'g', -1, 64)`, through
`internal/format`.

## NEEDS-90: an enum whose variant payload contains the enum

**Status:** blocking for `std/json.tw`. `Json`, cases `JArray` and `JObject`.

```
enum Json {
  ...
  JArray(Arr[Json]),
  JObject(Dict[Str, Json]),
}
```

`docs/self-hosting.md` section 1.2 specifies enums with payloads and `Arr[T]`
with an arbitrary element type, and NEEDS-72 asks for `T` to be allowed to be a
container. Neither says whether the recursion may close: whether a type may
appear inside its own payload, through a container.

It has to, and not only for JSON. `src/ast.tw` is already this shape (`Expr`
contains `Call` which contains `Arr[Expr]`) and gets away without a separate
entry because the recursion there goes through a struct. Both forms need the
same thing from the checker, which is that a type being defined is in scope
inside its own definition and that monomorphisation of `Arr[Json]` terminates
rather than instantiating forever.

Nothing exotic is wanted: the payload is behind a container, so the size is
finite and there is no infinite struct. The entry exists because the natural
reading of "monomorphized by the checker" is a worklist that would not
terminate here without a memo on the types already instantiated.

*Go bootstrap:* none. `internal/value.Value` is an interface, so recursion
through it never had to be decided.

## NEEDS-91: asking whether a path exists, without reading it

**Status:** open, a workaround is in place and it is absurd. `std/io.tw`
`exists`, `is_dir`.

`path_exists(Str) -> Bool`, or better a `stat` returning existence, kind and
size in one call.

The runtime surface in NEEDS-28 has `read_file`, `write_file` and `list_dir`
and nothing else, so `exists` is currently: try to read the whole file, and if
that fails, list the whole parent directory and look for the base name. It is
correct. It also reads a gigabyte to answer a yes-or-no question about a
gigabyte file, and lists a hundred thousand entries to answer one about a
directory with a hundred thousand entries.

The cost is the visible half. The invisible half is that a file which exists but
cannot be read reports differently depending on which branch answers, and a
directory that exists but cannot be listed reports false. A `stat` collapses all
of that into one answer.

*Go bootstrap:* `os.Stat`.

## NEEDS-92: removing a file, and a temporary directory to put one in

**Status:** open, low priority, and it is why `std/tests/io_test.tw` does not
test reading and writing.

`remove(Str) -> Res[Unit, Str]` and `temp_dir() -> Str`.

`docs/self-hosting.md` deliberately excludes directory operations, and for the
compiler that is right: it reads files, writes files and reports. A test suite
is the other caller, and it cannot write a fixture without leaving it behind. So
`io_test.tw` tests the path handling, which is where the bugs are, and does not
test the three-line wrappers over `read_file` and `write_file`, which is where a
runtime bug would be.

That is a real gap and it is recorded rather than papered over with a test that
writes into the source tree and hopes.

*Go bootstrap:* `os.Remove`, `os.MkdirTemp`, `testing.T.TempDir`.

## NEEDS-93: removing the last element of an `Arr`

**Status:** open, small, a workaround is in place. `std/io.tw` `normalize`.

`pop(a) -> Opt[T]`, or `truncate(a, n)`.

The primitive table has `arr_new`, `push`, indexed get and set, and `len`. There
is no way to make an array shorter, so `normalize` resolving a `..` component
rebuilds the whole stack one element shorter, which is O(n^2) over a path with
many of them. No real path has many of them, so this is a note rather than a
problem, and it is recorded because a stack is the natural shape for half a
dozen things in a compiler and every one of them will want the same operation.

The `Arr` is already growable, so the storage exists; this is the missing half
of `push`.

*Go bootstrap:* `s = s[:len(s)-1]`.

## NEEDS-94: a way to fail

**Status:** blocking, and worked around badly. `std/nn.tw` `init`,
`conv_init_as`.

`nn.init(strategy, nout, nin)` takes the initialisation strategy by name, so
that nobody gets Xavier when they meant He without being told. An unrecognised
name is a programming error and should stop the program with a message naming
the strategy and the caller. There is nothing in the language that stops a
program: no `error`, no `panic`, no `assert`, no way to return a failure that
cannot be ignored.

The workaround is `print` followed by a tensor of NaNs, chosen because the NaN
propagates into the first loss and is at least visible. It is still wrong in
both directions: the print goes to stdout in the middle of whatever else is
being printed, and the NaN surfaces one training step away from its cause, so
the message and the symptom are separated by everything the model did in
between.

Every module in this set has the same hole. `std/frame.tw` cannot say that a
column does not exist, `std/batch.tw` cannot say that a fold count exceeds the
row count, and `std/loss.tw` cannot say that a probability was passed where a
logit was wanted, which is the single most common mistake this library invites.

*Go bootstrap:* the interpreter's builtins return `(value.Value, error)` and the
error reaches the top with a source position attached. What is wanted is that
mechanism exposed to Twill, not a new one.

## NEEDS-95: a seeded random stream, or `permutation` taking a seed

**Status:** open, worked around, and the workaround has a side effect.
`std/batch.tw` `shuffled_indices`, `stratified_indices`,
`stratified_kfold_indices`.

Every split in `std/batch.tw` takes an explicit seed, because a split that
cannot be reconstructed makes the number measured on it unreproducible.
`permutation(n)` has no seed parameter, so the only way to honour that argument
is to call the global `seed(s)` first.

That works, and it moves the one random stream the whole program shares. A call
to `train_test_split` therefore changes every subsequent `randn`, so splitting
the data after initialising the model gives different weights than splitting
before it, for no reason the reader of that code could see. `stratified_indices`
makes it worse: it seeds once per class, so it consumes and resets the stream
several times in one call.

Either `permutation(n, seed)` and `randn(shape, seed)`, or a first-class
generator value that carries its own state and is threaded like the optimizer
state in `std/optim.tw`. The second is the better answer and the larger change;
the first would remove the surprise today.

*Go bootstrap:* `internal/interp` holds a package-level `*rand.Rand`. A
per-call seed is a second `rand.New(rand.NewSource(seed))` that is not stored.

## NEEDS-96: iteration that does not materialise

**Status:** open, and a real limit on dataset size. `std/batch.tw`
`epoch_batches`, `eval_batches`.

`epoch_batches` returns the whole epoch as a list of `[Xb, yb]` pairs. Every
batch of every epoch exists at once, which for a dataset that fits in memory
costs one extra copy of it, and for one that does not is simply the wrong
answer.

What is wanted is a generator: a function that can yield a value and be resumed,
or a lazy sequence with a `next`. Either would let a training loop pull one
batch at a time, and would also let `std/io` stream a file that does not fit in
memory, which is the same shape of problem.

A closure over mutable state is the workaround available today, and it is worse
than the list: `fn() { i = i + 1; ... }` has no way to say it is finished except
by a sentinel value the caller has to test for, which is exactly the pattern
that ends in reading one batch past the end.

*Go bootstrap:* none. The Go interpreter builds the same list.

## NEEDS-97: assigning to an element of a list

**Status:** open, worked around. `std/batch.tw` `stratified_kfold_indices`.

`xs[i] = v` is a syntax error. `append` is the only way to grow a list and there
is no way to replace an element of one, so an algorithm that fills k buckets by
dealing into them has to be turned inside out: `stratified_kfold_indices` makes
one pass per fold over the same shuffled per-class lists, deciding each time
whether an element belongs to the fold it is currently building. That is k times
the work for a result that one pass would produce, and the rewritten loop is
harder to read than the one it replaced.

Tensors have the same gap and a better excuse, since in-place mutation of a
tensor would have to be reflected on the tape. A list of indices carries no
gradient and has no such problem.

*Go bootstrap:* `[]value.Value` is a Go slice and assignment is assignment. The
restriction is the language's, not the runtime's.

## NEEDS-98: an empty record, and removing a field

**Status:** blocking. `std/frame.tw` has no `select`, `drop`, `rename` or
`from_columns` because of it, and `group_agg` cannot name its own output
columns.

`with_field(rec, name, value)` builds a record with a name known at run time,
which is exactly the right primitive, and it is unusable on its own because
there is nothing to start from. `{}` is a block and evaluates to unit, so there
is no empty record, and nothing removes a field, so a record cannot be narrowed
either. Every record therefore has to be born from a literal whose field names
are in the source text.

The consequence for a column-oriented table is severe. `select(df, names)` is
the most basic operation a table has and it cannot be written: it needs a record
whose fields come from a list. `drop` is `select` of the complement. `rename` is
`select` with one name changed. `group_agg` can compute its answer but has to
return it under the fixed names `key` and `value`, because it cannot name the
columns after the ones it grouped and aggregated.

Two primitives close it, and either alone would do:

    record()                   the empty record, so with_field can build any
    without_field(rec, name)   a copy without a field, so any record can be
                               narrowed

`record()` is the smaller and more general of the two: given it, `without_field`
is a fold over `columns` skipping one name.

*Go bootstrap:* `value.Record` is an ordered map. Both operations are three
lines each and neither has a design question in it.

## NEEDS-99: string concatenation

**Status:** specified; the implementation is still open, and until it lands this
pushes work onto callers. `std/frame.tw` `one_hot`. **The normative text is
`docs/language-guide.md`, Strings → Concatenation** for `+`, and Standard
library → `str` on a number for the round-tripping rendering this entry pairs
with it. Once both exist, `one_hot` builds `colour_0` itself and stops taking
the output names as an argument.

One-hot encoding a column called `colour` over the categories 0, 1, 2 should
produce columns called `colour_0`, `colour_1` and `colour_2`. There is no `+`
on strings, no `concat` for them, and no formatting function that takes a
string and a number and returns a string, so `one_hot` takes the output names
as an argument and the caller writes them out by hand. For a category with
thirty values that is thirty string literals at the call site, and every one of
them is a chance to get the order wrong, which produces a frame whose column
names disagree with its contents and nothing that would notice.

The same gap is why `std/metrics.tw` cannot label the rows of `describe`, and
why every diagnostic message in these modules is a `print` with several
arguments rather than one built string.

Wanted: `str_concat(a, b)` or `+` on strings, and a `str` that takes a number to
a decimal string that round-trips (NEEDS-89 asks for the second half of that
already).

*Go bootstrap:* Go strings concatenate with `+`. `internal/interp` already
formats numbers for `print`.

## NEEDS-100: enumerating and opening a GPU device

**Status:** open, and blocking for `src/gpu/` in its entirety. `src/gpu/device.tw`
`available`, `device_count`, `open`, `close`.

Read NEEDS-108 first. It says that none of NEEDS-100 through NEEDS-107 can be
implemented at all under the current rules, and that the entries exist so that
the requirement is a named list rather than an open question. Every entry in
this block is a signature, not a plan.

    gpu_available() -> Bool
    gpu_device_count() -> I64
    gpu_device_open(index: I64) -> Res[I64, Str]
    gpu_device_info(dev: I64, key: Str) -> Str
    gpu_device_info_i64(dev: I64, key: Str) -> I64
    gpu_device_close(dev: I64)

`gpu_available` must not fail and must not be an error condition. A machine with
no GPU driver is the normal case, and the whole graceful-degradation story is
that `available()` is false, every tensor stays on the host, and every answer is
unchanged. It is false when the driver library is absent, when it is present and
exports nothing usable, and when it reports zero devices.

The two `_info` forms exist because twill has no sum type at a primitive
boundary and a single accessor would have to return a string for `name` and an
integer for `compute_units`. The keys `src/gpu/device.tw` asks for are `name`,
`driver`, `compute_units`, `max_group`, `has_f64` and `local_bytes`. The last
three are not diagnostics: `max_group` and `local_bytes` decide whether the
tiled matmul can be launched at all, and a kernel that exceeds either does not
run slowly, it fails to launch.

Against OpenCL this is `clGetPlatformIDs`, `clGetDeviceIDs`, `clCreateContext`,
`clCreateCommandQueue` and `clGetDeviceInfo`, flattened so twill sees one list
of devices rather than a list of platforms each holding a list of devices. The
nesting buys nothing: `docs/gpu-feasibility.md` found two platforms with one
device each on the development machine, and code wanting the fastest device
would flatten it anyway.

*Go bootstrap:* none. `internal/tensor` is `[]float64` and goroutines and has no
concept of a device.

## NEEDS-101: allocating and freeing device memory

**Status:** open. `src/gpu/device.tw` `alloc`, `free`; `src/gpu/buffer.tw`
`alloc_with_eviction`.

    gpu_alloc(dev: I64, elements: I64) -> Res[I64, Str]
    gpu_free(buf: I64)

Sized in F64 elements and not bytes, because every caller in `src/gpu/` counts
in elements and a units mismatch at this boundary reads as a wrong answer rather
than as a crash.

`gpu_alloc` must return `Err` on an out-of-memory rather than abort, and this is
the one primitive whose failure mode is designed around. The card in the
development machine has 8GB shared with the display, so a long run exhausting it
is expected rather than exceptional. `src/gpu/buffer.tw` catches the `Err`,
evicts device copies whose host copy is still valid, retries once, and then
falls back to the CPU with an identical answer. A primitive that aborted would
turn memory pressure into a failed program.

`gpu_free` returns nothing. A failure to free is not something a caller can act
on, and a `Res` here would put a `?` on every cleanup path to serve no decision.

Against OpenCL: `clCreateBuffer` with `CL_MEM_READ_WRITE`, and
`clReleaseMemObject`.

*Go bootstrap:* none.

## NEEDS-102: moving numbers to and from a device

**Status:** open. `src/gpu/device.tw` `write`, `read`, `copy`.

    gpu_write(buf: I64, dst_off: I64, src: Arr[F64]) -> Res[Unit, Str]
    gpu_read(buf: I64, src_off: I64, n: I64)         -> Res[Arr[F64], Str]
    gpu_copy(dst: I64, dst_off: I64, src: I64, src_off: I64, n: I64)
                                                     -> Res[Unit, Str]

All three blocking, in the first version. `docs/gpu.md` section 3 argues that a
non-blocking queue reports an error at a point unrelated to the op that caused
it, and that debugging a numerical difference is hard enough with the error
attached to the right line.

`gpu_read` allocates and returns a fresh `Arr[F64]` rather than filling one the
caller supplies, because twill has no way to hand out a writable window into an
existing `Arr` and pretending otherwise would put an aliasing rule into the one
place in the codebase that cannot check it.

`gpu_copy` is device to device and is not a convenience. Without it, taking a
row out of a device tensor means reading it down and writing it back up, which
is two of the boundary crossings `docs/gpu-feasibility.md` measured at roughly
80us each, to move data that never needed to leave. It is what keeps slice,
concat, index and split resident.

Note what is deliberately *not* here: an integer transfer. The elementwise
kernels need shapes and strides on the device, and those ride as `F64` and are
cast back in the kernel. A shape cannot exceed 2^53 without the tensor exceeding
any device this will run on, so the round trip is exact, and the ugliness buys
one fewer entry on this list. See `src/gpu/buffer.tw` `meta_buffer`.

Against OpenCL: `clEnqueueWriteBuffer` and `clEnqueueReadBuffer` with
`blocking = CL_TRUE`, and `clEnqueueCopyBuffer`.

*Go bootstrap:* none. A slice is already where the CPU wants it.

## NEEDS-103: compiling a kernel from source at run time

**Status:** open. `src/gpu/device.tw` `build`, `kernel`.

    gpu_program_build(dev: I64, source: Str, options: Str) -> Res[I64, Str]
    gpu_kernel(program: I64, name: Str)                    -> Res[I64, Str]

Run-time compilation from source text is the property that made OpenCL the
recommendation in `docs/gpu.md` over Vulkan, which consumes SPIR-V and would
mean either shipping precompiled binary blobs built by a toolchain that is not
present, or writing a SPIR-V emitter. Compiling from source means the kernels
are readable text in the repository, there is no build step, nothing is added to
the release matrix, and a kernel can be specialised on the shapes it is about to
run. `src/gpu/matmul.tw` uses that last property to bake its tile size in as a
compile-time constant, which is what lets the compiler unroll the inner loop and
size the local arrays statically.

The `Err` of `gpu_program_build` is the most important error message in the
backend and it must carry the driver's build log verbatim. A kernel that fails
to build fails on somebody else's driver, on hardware nobody developing twill
owns, and the log is the only evidence that will ever exist.

`options` is the compile options string, and what is absent from it is the
subject of `docs/gpu.md` section 5 rule 3. It is built in `src/gpu/source.tw` so
there is exactly one of it.

Against OpenCL: `clCreateProgramWithSource`, `clBuildProgram`,
`clGetProgramBuildInfo` for the log, and `clCreateKernel`.

*Go bootstrap:* none.

## NEEDS-104: binding kernel arguments

**Status:** open. `src/gpu/device.tw` `arg_buffer`, `arg_i64`, `arg_f64`,
`arg_local`.

    gpu_set_arg_buffer(kernel: I64, index: I64, buf: I64)  -> Res[Unit, Str]
    gpu_set_arg_i64(kernel: I64, index: I64, v: I64)       -> Res[Unit, Str]
    gpu_set_arg_f64(kernel: I64, index: I64, v: F64)       -> Res[Unit, Str]
    gpu_set_arg_local(kernel: I64, index: I64, bytes: I64) -> Res[Unit, Str]

Four setters and not one. A kernel argument is typed on the device side, and
passing an integer where a buffer was expected is undefined rather than an
error. Twill has no variadic call and no way to describe a heterogeneous
argument list, so the alternative is an encoding, and an encoding here would be
a second place for the two sides' types to disagree.

`arg_local` reserves work-group local memory for an argument the kernel declares
`__local` with no size. Only the tiled matmul uses it, and it is on the list
rather than folded away because a matmul without local-memory staging is the
untiled version, which is several times slower and is the reason a GPU is being
considered at all.

Against OpenCL: `clSetKernelArg`, four times over, with `arg_local` passing a
size and a null pointer.

*Go bootstrap:* none.

## NEEDS-105: launching a kernel

**Status:** open. `src/gpu/device.tw` `launch`.

    gpu_launch(dev: I64, kernel: I64, global: Arr[I64], local: Arr[I64])
        -> Res[Unit, Str]

`global` is the total number of work-items per dimension, 1 to 3 dimensions.
`local` is the work-group shape, or empty to let the driver choose. Empty is the
default everywhere except the tiled matmul, which needs a specific group shape
for its local-memory staging to be *correct* and not merely fast: a barrier that
only some work-items in a group reach is undefined behaviour.

Note what `gpu_launch` does not take: a stream, an event, or a dependency. There
is one queue and every launch is followed by a synchronise. That is the first
thing to change once the answers are trusted, which is why NEEDS-106 is a
separate entry rather than folded into this one.

Against OpenCL: `clEnqueueNDRangeKernel`.

*Go bootstrap:* the nearest analogue is `runChunks` in
`internal/tensor/parallel.go`, which splits an index range across goroutines.
The shape of the idea is the same and nothing else about it is.

## NEEDS-106: synchronising with a device

**Status:** open. `src/gpu/device.tw` `finish`.

    gpu_finish(dev: I64) -> Res[Unit, Str]

Blocks until every command queued on the device has completed. This is where a
kernel's error surfaces, because an enqueue that returned `Ok` has only been
accepted and not run.

It is its own entry rather than part of NEEDS-105 for a forward-looking reason.
The first version calls it after every launch, which throws away the latency
hiding a deep queue would give. Letting the queue run ahead is the single
largest easy win left in the design once the answers are trusted, and it is only
possible if launch and synchronise are separable.

Against OpenCL: `clFinish`.

*Go bootstrap:* a `sync.WaitGroup`, in the sense that both wait.

## NEEDS-107: loading a shared library and resolving a symbol at run time

**Status:** open, and the mechanism NEEDS-100 through NEEDS-106 all rest on.

The six entries above are signatures. This one is the thing that makes any of
them reachable: a way to open a shared library by name at run time and look up a
symbol in it, then call through the resulting pointer with a described
signature.

`docs/gpu-feasibility.md` established that this is the whole dependency story
for OpenCL. `OpenCL.dll` is in `System32` on the development machine because the
driver installed it, both GPUs register ICDs, and a host program that resolves
the loader with `LoadLibrary` and `GetProcAddress` and declares the dozen entry
points it needs compiled with no headers and no SDK and ran on both cards. So
what is wanted is not an SDK binding. It is `LoadLibrary` plus `GetProcAddress`,
and their equivalents elsewhere, plus a calling convention.

That is deliberately more general than a GPU. It is a foreign function
interface, and every consumer of one that twill might ever have goes through it.
It is recorded here because the GPU backend is the first concrete thing that
needs it and therefore the first thing that can say precisely what shape it must
have: pointer-sized handles, `Arr[F64]` passed as a base pointer and a length,
`I64` and `F64` scalars, and a return that is either an integer status or a
handle.

The alternative that avoids it entirely is native code compiled into the
runtime, which is the same requirement wearing a different hat and is the
subject of NEEDS-108.

*Go bootstrap:* `internal/interp/builtins.go` dispatches on a name into Go
functions in the same binary, which needs no FFI because there is no foreign
side. That is exactly the property the GPU backend does not have.

## NEEDS-108: there is nowhere for a native layer to live

**Status:** open, and it is not a language feature. It is the reason NEEDS-100
through NEEDS-107 cannot be started.

Stated plainly, because the rest of `src/gpu/` and `docs/gpu.md` are written as
though it were solved and a reader should not have to infer this from their
silence:

**A GPU backend cannot exist without a foreign function interface or native code
of some kind. Under the current no-Go rule, that layer has nowhere to live.**

Nothing in `src/gpu/` closes this and no amount of further twill would. Every
kernel in `src/gpu/source.tw` is text that a driver has to compile, and every
function in `src/gpu/device.tw` is a call into a library that twill has no way
to call. The design is complete and unrunnable, and those are two separate
facts.

The options, none of which is a language feature:

1. **An FFI, which is NEEDS-107.** The most general answer and the largest. It
   gives twill a way to call anything, which is a much bigger decision than
   "should there be a GPU backend" and should not be made as a side effect of
   one.
2. **Native code in the runtime.** Whatever eventually executes twill has to be
   written in something, and that something can link the loader directly. This
   is what `internal/` does today for `math.Exp`, which NEEDS-68 already treats
   as a native primitive rather than a foreign call. Under that framing the
   fifteen entries above are fifteen more native primitives alongside `f64_exp`,
   and the question stops being "how does twill call out" and becomes "what is
   on the primitive table". That is the smaller and more likely answer.
3. **Neither, for now.** Which is where things stand, and which
   `docs/gpu-feasibility.md` recommends on independent grounds: settle f32
   first, add the matmul benchmarks at N=256, 512 and 1024 that the repository
   does not have, and find a real twill program that is matmul-bound at 256x256
   or larger before building any of this.

The value of writing the list anyway is that the requirement is now bounded.
Fifteen symbols out of a library that is already installed on any machine with a
GPU, resolved at run time, with nothing added to the build and nothing added to
the release matrix. That is a small enough number to argue about honestly, which
is the whole point of counting it.

*Go bootstrap:* it links `internal/tensor` directly, which is the option-2
answer taken without anyone having to decide it.

## NEEDS-109: `reduce_all` disagrees with the bootstrap above 8192 elements

**Status:** open, and it is a correctness divergence rather than a missing
feature. `src/tensor.tw` `reduce_all`; `internal/tensor/parallel.go`
`parallelSum`; `src/gpu/reduce.tw` `whole_tensor_sum`.

Found while designing the GPU reductions, and recorded rather than fixed because
`src/tensor.tw` was owned by another change at the time.

`internal/tensor/parallel.go` sums a whole tensor in two forms. Below
`minParallel = 8192` it is a plain running sum. At or above it, fixed
4096-element blocks are summed independently and their partials combined in
block order, and the comment is explicit that the block size is fixed rather
than derived from the core count so that "the result is the same on any number
of cores".

`src/tensor.tw` `reduce_all` is a plain running sum at every size, with no
blocking. So for `n >= 8192` the twill reference and the Go bootstrap produce
different last bits for the same input, today, with no GPU anywhere near it.

Three implementations cannot all be right. The Go form is the one to adopt,
because a fixed block size is a *specification* rather than an implementation
detail: it pins the answer, it is reproducible on any hardware, and it is
parallelisable, which the plain running sum is not. `src/gpu/reduce.tw` follows
the Go form for exactly that reason, and is therefore bit-identical to the
bootstrap and not to `src/tensor.tw` until this is closed.

This matters more than a last-bit divergence usually would, because `testdata/`
compares output byte for byte after a canonical float rendering (NEEDS-29). A
divergence in the last bits of a sum over 8192 or more elements is a test
failure, not a curiosity.

*Go bootstrap:* it is the correct one here. `parallelSum` in
`internal/tensor/parallel.go` is the text to port.
