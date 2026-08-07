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

## NEEDS-15 — lexer divergence: non-ASCII whitespace in comments

**Status:** known, accepted divergence, **confirmed by differential test**.
`src/lex.tw:466` (`is_space`).

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

---

## Found by differential testing of the lexer

The three entries below came out of running `src/lex.tw` against
`internal/lexer/lexer.go` over 385 corpus files and 4,000 fuzzer cases. See
"Verification" at the end of this file.

## NEEDS-32 — Go-compatible `%q` for a non-printable or non-ASCII character

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

## NEEDS-33 — the Go bootstrap panics on a trailing backslash at end of file

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

**Status:** blocking. Every renderer builds its output by concatenation.

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

**Status:** blocking for `src/cli/banner.tw` and `src/cli/tensor.tw`.

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

**Status:** blocking for `src/term/frame.tw`, `src/cli/spinner.tw`,
`src/cli/progress.tw`, `src/cli/repl.tw`.

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

**Status:** blocking, and it is a semantics question rather than a missing
operator.

`/` and `%` exist, but on tensors they are float operations. On `I64` they must
truncate toward zero, and `%` must take the sign of the dividend. The 256-colour
quantisation in `src/term/color.tw`, the braille packing in
`src/cli/banner.tw`, and the eighth-block arithmetic in `src/cli/progress.tw`
are all exact integer arithmetic and are wrong under any other rounding.

Also needed: division by zero on `I64` as an error value rather than a panic,
which section 1.2 already specifies.

## NEEDS-45 - `str()` on `I64`

**Status:** blocking.

`str(n)` for an `I64` must produce the digits with no decimal point and no
exponent. Today `str` on a scalar goes through the tensor printer, and a
trailing `.0` would land in every line number, every column count and every axis
index in every diagnostic.

*Reaches for it:* everywhere. `src/cli/diagnostic.tw` alone uses it for the
line, the column and the gutter width.

## NEEDS-46 - `Str` equality must survive the `Str` rewrite

**Status:** not a new feature. Recorded as a constraint.

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

## NEEDS-34 — the systems-mode checker policy

**Status:** open, and it is the design decision, not a coding task.

`docs/self-hosting.md` section 1.3: in systems mode a `TUnknown` surviving to
the end of inference is a diagnostic rather than silence. `src/check.tw`
implements the numeric policy, because that is what `testdata/` was generated
against and what the diagnostics are compared to. The systems policy is a second
policy over the same lattice and it is what would let the checker check `src/`
itself.

Until it exists, the self-hosted compiler is not type-checked by its own
checker, which is an uncomfortable place to be and worth naming.

## NEEDS-35 — an out-of-range axis in `transpose`

**Status:** open, low priority. `src/check.tw` `transpose_result`.

An axis outside the tensor's rank makes the result unknown rather than a
diagnostic, matching `internal/checker/checker.go`. Every other axis-taking
builtin reports it through `report_axis`. Fixing this means changing the Go side
too, or the diagnostics diverge.

## NEEDS-36 — the import resolver

**Status:** blocking for `src/eval.tw`. `exec_import`.

The policy is written out in the comment there: relative to the importing file,
`std/` resolves to the embedded library unless `TWILL_STD` overrides, `.ra` is
refused outright, a module is evaluated once and cached, and an aliased import
lands in a record while an unaliased one lands unqualified.

What is missing is file reading (NEEDS-28) and a way to reach the embedded
standard library from twill, which the bootstrap does with `go:embed`.

## NEEDS-37 — one builtin table, not two

**Status:** open, tidiness. `src/check.tw` and `src/eval.tw`.

The checker's table is what the diagnostics are compared against; the
evaluator's would be what the dispatch uses. They are separate today because
merging them before the dispatch exists would be guessing at its shape.

## NEEDS-38 — the formatter

**Status:** not started. `src/main.tw` `cmd_fmt`.

`internal/format/format.go` has not been ported. `twill fmt` reports that rather
than silently doing nothing.

One rule has to survive the port because it is the one people notice: `fmt
--write` never renames a file. It refuses a `.ra` file and leaves it alone.

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
