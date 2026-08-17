package interp_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/twill-lang/twill/internal/interp"
)

// The language changes that went into 1.5. Each one came from a repository in
// the ecosystem that could not run without it, so each test here is written the
// way that repository wrote the code, not the way a language test usually reads.

// run evaluates src and returns its printed lines.
func run15(t *testing.T, src string) []string {
	t.Helper()
	dir := t.TempDir()
	return runFile(t, dir, src)
}

func expectLines(t *testing.T, got []string, want ...string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("output = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("line %d = %q, want %q", i, got[i], want[i])
		}
	}
}

// A line opening with `+` or `-` continues the statement above it when it is
// indented past that statement, and starts a new one when it lines up.
func TestLeadingOperatorContinuation(t *testing.T) {
	out := run15(t, `let a = 1.0
  + 2.0
  - 0.5
print(a)
let s = "x"
  + "y"
print(s)
let g = 5.0
-3.0
print(g)
`)
	expectLines(t, out, "2.5", "xy", "5")
}

// The continuation rule also has to leave a line that opens with a bitwise call
// alone: `xor(...)` at statement indentation is a new statement.
func TestLeadingBitwiseCallStartsStatement(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let a = 6
  xor(a, 3)
  print(a)
}
`)
	expectLines(t, out, "6")
}

// band/bor/xor/shl/shr are infix and called, and mean the same either way.
// `and` stays boolean infix, which is why band exists at all.
func TestBitwiseInfixOperators(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let x = 1000
  print(x shr 3)
  print(x shl 2)
  print(x xor 255)
  print((x shr 4) band 15)
  print(x bor 3)
  print(shr(x, 3))
  print(band(x, 255))
  print(true and false)
}
`)
	expectLines(t, out, "125", "4000", "791", "14", "1003", "125", "232", "false")
}

// An empty literal at a container annotation builds that container.
func TestEmptyContainerFollowsAnnotation(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let d: Dict[Str, I64] = {}
  dict_set(d, "k", 7)
  print(dict_must(d, "k"))
  let xs: Arr[Str] = []
  print(len(arr_push(xs, "q")))
}
`)
	expectLines(t, out, "7", "1")
}

// A dictionary is subscripted by its key, read and written.
func TestDictSubscript(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let d: Dict[Str, I64] = {}
  d["a"] = 1
  d["a"] = d["a"] + 5
  print(d["a"])
}
`)
	expectLines(t, out, "6")
}

// xs.push(v) is push(xs, v) when the target has no such field.
func TestUniformCallSyntax(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let xs: Arr[F64] = []
  xs.push(1.0)
  xs.push(2.0)
  print(xs.len())
  print("hello".len())
}
`)
	expectLines(t, out, "2", "5")
}

// A record field still wins over the uniform-call fallback, so a namespace or a
// record of closures keeps the field call it means.
func TestUniformCallDoesNotShadowRecordField(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let r = { len: 99 }
  print(r.len)
}
`)
	expectLines(t, out, "99")
}

// Opt.Some(x) and Opt.None are the bare variants, in expressions and patterns.
func TestQualifiedVariants(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let a = Opt.Some(3)
  match a {
    Opt.Some(v) => print(v),
    Opt.None => print("none"),
  }
  match a {
    Some(v) => print(v),
    None => print("none"),
  }
  print(Opt.None)
}
`)
	expectLines(t, out, "3", "3", "None")
}

// Fn(T) -> R is the function type, alongside fn(T) -> R, in every annotation
// position: parameter, return, struct field and let.
func TestCapitalisedFunctionType(t *testing.T) {
	out := run15(t, `mode systems
struct P {
  f: Fn(I64) -> I64,
  g: fn(I64) -> I64,
}
fn inc(n: I64) -> I64 = n + 1
fn ap(f: Fn(I64) -> I64, x: I64) -> I64 = f(x)
fn main() {
  print(ap(inc, 4))
}
`)
	expectLines(t, out, "5")
}

// arr_push hands the list back, so it reads as an expression.
func TestArrPushReturnsTheList(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let out: Arr[Str] = []
  out = arr_push(out, "a")
  out = arr_push(out, "b")
  print(len(out))
  print(out[1])
}
`)
	expectLines(t, out, "2", "b")
}

// exit(n) stops with a status and reports it as an ExitError rather than a
// fault, which is what lets a test harness fail a red suite.
func TestExitCarriesItsStatus(t *testing.T) {
	dir := t.TempDir()
	main := writeModule(t, dir, "main", `mode systems
fn main() {
  print("before")
  exit(3)
  print("after")
}
`)
	var out []string
	ip := interp.New(func(s string) { out = append(out, s) })
	_, _, err := ip.RunFileMain(main, nil)
	var ex *interp.ExitError
	if !errors.As(err, &ex) {
		t.Fatalf("err = %v, want an ExitError", err)
	}
	if ex.Code != 3 {
		t.Fatalf("exit code = %d, want 3", ex.Code)
	}
	expectLines(t, out, "before")
}

// arr_of_tensor copies a rank-1 tensor into an Arr[F64].
func TestArrOfTensor(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let xs = arr_of_tensor(tensor([1.0, 2.5, 3.0]))
  print(len(xs))
  print(xs[1])
}
`)
	expectLines(t, out, "3", "2.5")
}

// std/hash is a SHA-256 that agrees with the standard vectors. It is the
// regression test for two bugs at once: `not` where `bnot` was meant, and a
// 32-bit rotate that overflowed the 53 bits an f64-backed I64 holds.
func TestStdHashMatchesTheStandardVectors(t *testing.T) {
	out := run15(t, `import "std/hash" as hash
print(hash.hash_str(""))
print(hash.hash_str("abc"))
`)
	expectLines(t, out,
		"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
}

// The terminal layer is reachable as a standard-library module, and the library
// carries one level of grouping to hold it.
func TestStdTermIsImportable(t *testing.T) {
	out := run15(t, `import "std/term/caps" as cp
import "std/term/theme" as th
import "std/term/box" as box
print("ok")
`)
	expectLines(t, out, "ok")
}

// A std path may not walk out of the library.
func TestStdPathCannotEscape(t *testing.T) {
	dir := t.TempDir()
	main := writeModule(t, dir, "main", "import \"std/../secret\"\n")
	ip := interp.New(func(string) {})
	_, _, err := ip.RunFileMain(main, nil)
	if err == nil {
		t.Fatal("expected an error for a std path with ..")
	}
	if !strings.Contains(err.Error(), "not a standard-library module name") {
		t.Fatalf("err = %v, want it refused as a module name", err)
	}
}

// A file's own function beats a builtin of the same name, whichever side of the
// declaration the call is written on. Before 1.5 only the call below it
// resolved, and the one above was checked against the builtin's arity.
func TestDeclarationShadowsBuiltinBothWays(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  print(read_csv("x", 3))
  print(below())
}
fn read_csv(p: Str, n: I64) -> I64 = n
fn below() -> I64 = read_csv("y", 7)
`)
	expectLines(t, out, "3", "7")
}

// `//` truncates toward zero; `/` stays exact float division. The two integer
// idioms the ecosystem kept getting wrong: a ceiling division and a midpoint.
func TestIntegerDivision(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  print(314 // 100)
  print(7 // 2)
  print(-7 // 2)
  print(7 / 2)
  print((10 + 3) // 4)
  print(314 % 100)
}
`)
	expectLines(t, out, "3", "3", "-3", "3.5", "3", "14")
}

func TestIntegerDivisionByZeroIsAnError(t *testing.T) {
	dir := t.TempDir()
	main := writeModule(t, dir, "main", `mode systems
fn main() { print(1 // 0) }
`)
	ip := interp.New(func(string) {})
	if _, _, err := ip.RunFileMain(main, nil); err == nil ||
		!strings.Contains(err.Error(), "integer division by zero") {
		t.Fatalf("err = %v, want an integer division by zero", err)
	}
}

// A `-> I64` return truncates its value, the same rule `let n: I64 = ...`
// applies to a bound one. Before this the two annotations disagreed.
func TestI64ReturnTruncatesLikeI64Binding(t *testing.T) {
	out := run15(t, `mode systems
fn ceil_div(n: I64, b: I64) -> I64 { (n + b - 1) / b }
fn half(n: I64) -> I64 = n / 2
fn frac(x: F64) -> F64 = x / 2.0
fn main() {
  let bound: I64 = (10 + 3) / 4
  print(bound)
  print(ceil_div(10, 4))
  print(ceil_div(0, 4))
  print(half(7))
  print(half(-7))
  print(frac(7.0))
}
`)
	expectLines(t, out, "3", "3", "0", "3", "-3", "3.5")
}

// An f64's bit pattern does not survive a round trip through an f64-backed I64,
// so the halves are exposed instead and each is exact. This is what makes a
// bit-for-bit serialisation format possible in twill at all.
func TestF64HalvesRoundTripExactly(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let xs = [0.1, -2.5, 0.0, 1.0e300, 3.0e-310]
  let i = 0
  while i < len(xs) {
    let x = xs[i]
    print(f64_from_halves(f64_bits_hi(x), f64_bits_lo(x)) == x)
    i = i + 1
  }
}
`)
	expectLines(t, out, "true", "true", "true", "true", "true")
}

// Strings order by byte, so a list of names can be sorted without each caller
// writing its own compare_str. Three repositories had done exactly that.
func TestStringComparisonOperators(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  print("apple" < "banana")
  print("b" > "a")
  print("ab" < "abc")
  print("Z" < "a")
}
`)
	expectLines(t, out, "true", "true", "true", "true")
}

// A cast applies to a scalar as much as to a tensor: a plain number is carried
// as a Num rather than a rank-0 tensor, and that is an internal distinction a
// program cannot see.
func TestCastAppliesToAScalar(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  print(dtype(tensor([1.5, 2.5]).to(bf16)))
  print(dtype((3.5).to(f32)))
  print(dtype(tensor([1.5]).to(bf16).to(f64)))
}
`)
	expectLines(t, out, "bf16", "f32", "f64")
}

// all_finite is the overflow check a loss scale is raised against. It reaches
// into a list or a record, because a gradient arrives as a tree.
func TestAllFinite(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  print(all_finite(tensor([1.0, 2.0])))
  print(all_finite(tensor([1.0, 0.0 / 0.0])))
  print(all_finite(tensor([1.0 / 0.0])))
  let xs: Arr[Tensor] = []
  push(xs, tensor([1.0]))
  print(all_finite(xs))
  push(xs, tensor([0.0 / 0.0]))
  print(all_finite(xs))
}
`)
	expectLines(t, out, "true", "false", "false", "true", "false")
}

// numel is the product of the shape, at any rank; arr_of_tensor copies the
// elements out in row-major order, also at any rank.
func TestNumelAndArrOfTensorAtAnyRank(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let m = tensor([[1.0, 2.0], [3.0, 4.0]])
  print(numel(m))
  let xs = arr_of_tensor(m)
  print(len(xs))
  print(xs[2])
}
`)
	expectLines(t, out, "4", "4", "3")
}

// A typed literal reads its struct's declared field types, so an empty literal
// at a container-typed field builds that container.
func TestEmptyContainerAtAStructField(t *testing.T) {
	out := run15(t, `mode systems
struct Cat { versions: Dict[Str, Str], names: Arr[Str] }
fn main() {
  let c = Cat { versions: {}, names: [] }
  c.versions["a"] = "1.0"
  print(c.versions["a"])
  print(len(arr_push(c.names, "x")))
}
`)
	expectLines(t, out, "1.0", "1")
}

// file_size answers in bytes, and -1 rather than raising when the path is not
// there, so a caller watching a file for changes can just compare.
func TestFileSize(t *testing.T) {
	dir := t.TempDir()
	writeModule(t, dir, "data", "hello")
	out := runFile(t, dir, `mode systems
fn main() {
  print(file_size("data.tw"))
  print(file_size("no_such_file.tw"))
}
`)
	expectLines(t, out, "5", "-1")
}

// An annotation says which container is meant, and that now covers the two
// cases a bracket literal cannot decide for itself.
func TestAnnotationChoosesTheContainer(t *testing.T) {
	out := run15(t, `mode systems
fn row(v: F64) -> Tensor = [v, v, v]
fn main() {
  # A bracket of numeric literals is a tensor; a bracket of expressions is a
  # list. The annotation settles both.
  let want: Arr[I64] = [1]
  print(len(arr_push(want, 5)))
  print(shape(row(2.0))[0])
  let t: Tensor = [1.0, 2.0]
  print(numel(t))
}
`)
	expectLines(t, out, "2", "3", "2")
}

// A list that is not numeric is left alone at a Tensor annotation: it is a list
// the caller built on purpose.
func TestNonNumericListIsNotCoercedToATensor(t *testing.T) {
	out := run15(t, `mode systems
fn names() -> Tensor = ["a", "b"]
fn main() { print(len(names())) }
`)
	expectLines(t, out, "2")
}

// Independent generator streams: reproducible from a seed, and independent
// across seeds. std/random is built on these.
func TestRngStreamsAreIndependentAndReproducible(t *testing.T) {
	out := run15(t, `mode systems
fn main() {
  let a = rng_open(1)
  let b = rng_open(1)
  let c = rng_open(2)
  print(rng_u53(a) == rng_u53(b))
  print(rng_u53(a) == rng_u53(c))
  print(rng_f64(a) < 1.0)
  rng_close(a)
}
`)
	expectLines(t, out, "true", "false", "true")
}

// std/random draws from those streams and is statistically sound, which the
// twill implementation it replaced was not: that one returned a constant.
func TestStdRandomIsNotConstant(t *testing.T) {
	out := run15(t, `import "std/random" as rd
let r = rd.new_rng(42)
let n = 4000
let s = 0.0
let distinct = 0
let seen = 0.0
let i = 0
while i < n {
  let u = rd.uniform(r)
  s = s + u
  if u != seen { distinct = distinct + 1 }
  seen = u
  i = i + 1
}
print(distinct == n)
print(s / n > 0.45 and s / n < 0.55)
`)
	expectLines(t, out, "true", "true")
}
