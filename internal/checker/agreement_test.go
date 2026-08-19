package checker_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/twill-lang/twill/internal/checker"
	"github.com/twill-lang/twill/internal/interp"
	"github.com/twill-lang/twill/internal/parser"
)

// Nothing in this repository used to check the checker against the runtime on
// the same program, and that gap is where checker bugs shipped from: the
// interpreter suite calls interp.Run directly, which never runs the checker, so
// no amount of interpreter testing can see a checker that is wrong. Five of the
// six defects fixed in 1.6.2 are in this file's reach.
//
// The rule being tested is the checker's own contract, from the package
// comment: report a mismatch only when it is certain, and stay quiet otherwise.
// That gives two ways to be wrong, and both are worth a name.
//
//	A FALSE POSITIVE is a diagnostic on a program that runs correctly. It is the
//	worse kind: it makes the checker a liar, and a liar gets turned off.
//
//	A FALSE NEGATIVE is silence on a program that then fails at run time with
//	something the checker had the facts to prove.

// runs evaluates a program with the checker out of the way, reporting whether
// it completed and what it printed.
func runsCleanly(t *testing.T, src string) (out string, err error) {
	t.Helper()
	var sb strings.Builder
	ip := interp.New(func(s string) { sb.WriteString(s) })
	// A program that reaches the filesystem or the clock is not this test's
	// business; everything here is pure.
	_, err = ip.Run(src)
	return sb.String(), err
}

// noFalsePositive asserts that a program which runs correctly draws no
// diagnostic.
func noFalsePositive(t *testing.T, src string) {
	t.Helper()
	if _, err := runsCleanly(t, src); err != nil {
		t.Fatalf("this program was supposed to run: %v\nsource:\n%s", err, src)
	}
	if diags := diagnostics(t, src); len(diags) != 0 {
		t.Errorf("FALSE POSITIVE: the program runs, but check says %v\nsource:\n%s", diags, src)
	}
}

// noFalseNegative asserts that a program which fails at run time was reported
// before it ran.
func noFalseNegative(t *testing.T, src string) {
	t.Helper()
	if _, err := runsCleanly(t, src); err == nil {
		t.Fatalf("this program was supposed to fail at run time\nsource:\n%s", src)
	}
	if diags := diagnostics(t, src); len(diags) == 0 {
		t.Errorf("FALSE NEGATIVE: the program fails at run time and check is silent\nsource:\n%s", src)
	}
}

// Programs that run. Every one of these was, or could have been, a false
// positive; the `Bool` case was one in 1.6.1, in the default mode.
func TestCheckerIsQuietOnProgramsThatRun(t *testing.T) {
	for _, src := range []string{
		"let b: Bool = true\nprint(b)",
		"let n: I64 = 3\nprint(n)",
		"let s: Str = \"x\"\nprint(s)",
		"let f: F64 = 1.5\nprint(f)",
		"let n: I64 = 9223372036854775807\nprint(n)",
		"let n: I64 = -9223372036854775808\nprint(n)",
		"let k: I64 = 2.5e3\nprint(k)",
		"fn name(x: Str) -> Str = x\nprint(name(\"a\"))",
		"print(1.0 < 2.0)",
		"print(greater([1.0, 2.0], 0.0))",
		"let a = [[1.0, 2.0], [3.0, 4.0]]\nprint(a @ a)",
		"fn mm(a: [2, 3], b: [3, 2]) -> [2, 2] = a @ b\nprint(mm(zeros(2, 3), zeros(3, 2)))",
		"print(sum(zeros(2, 3), 1))",
		"print(reshape(zeros(2, 3), 3, 2))",
		"print(concat([zeros(2), ones(3)], 0))",
		"unit USD\nlet p: USD = 3.0\nprint(p + p)",
	} {
		noFalsePositive(t, src)
	}
}

// Programs that fail at run time, where the checker had the facts. Each of
// these was a false negative at some point.
func TestCheckerReportsWhatFailsAtRuntime(t *testing.T) {
	for _, src := range []string{
		// Ordering a tensor: the `where(A > 0, ...)` masking idiom.
		"print([1.0, 2.0] > 0.0)",
		"let a = [[1.0, 2.0], [3.0, 4.0]]\nprint(where(a > 0.0, a, a))",
		// Shapes the checker can prove.
		"print(zeros(2, 3) @ zeros(2, 2))",
		"print(zeros(2, 3) + zeros(4))",
		"print(reshape(zeros(2, 3), 4, 2))",
		// A name that is not defined.
		"print(nope)",
	} {
		noFalseNegative(t, src)
	}
}

// A function that falls off its end hands back `()`, and the caller was
// promised something else. The checker skipped this whole case.
func TestAFunctionThatFallsOffItsEndIsCaught(t *testing.T) {
	const src = "mode systems\nfn name(b: Bool) -> Str {\n  if b { \"yes\" }\n}\nprint(name(false))"
	diags := diagnostics(t, src)
	if len(diags) == 0 {
		t.Fatal("FALSE NEGATIVE: a function declaring Str returns () and check is silent")
	}
	if !strings.Contains(diags[0].Msg, "returns Unit") {
		t.Errorf("msg = %q, want it to name the Unit return", diags[0].Msg)
	}
}

// The corpus, checked in bulk. Every file the repository ships should draw no
// diagnostic, since these are the programs the tests and examples run. It is a
// blunt instrument and it is the one that catches a checker change that starts
// reporting something ordinary.
func TestTheRepositorysOwnSourcesCheckClean(t *testing.T) {
	roots := []string{
		filepath.Join("..", "..", "std"),
		filepath.Join("..", "..", "examples"),
		filepath.Join("..", "..", "src"),
	}
	checked := 0
	for _, root := range roots {
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() || !strings.HasSuffix(path, ".tw") {
				return nil
			}
			src, readErr := os.ReadFile(path)
			if readErr != nil {
				return nil
			}
			prog, perr := parser.Parse(string(src))
			if perr != nil {
				t.Errorf("%s does not parse: %v", path, perr)
				return nil
			}
			// CheckFile, so an enum reached through an import resolves.
			if diags := checker.CheckFile(prog, path); len(diags) != 0 {
				t.Errorf("%s draws %d diagnostic(s): %v", path, len(diags), diags[0])
			}
			checked++
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if checked < 40 {
		t.Fatalf("only %d files checked; the walk is not finding the sources", checked)
	}
}

// The five findings of the 1.6.4 soundness pass, each a program the checker
// had every fact it needed to judge and did not.

// `[]` evaluates to an empty list, not a tensor of shape [0]. Typing it as a
// tensor made the checker confidently wrong at the dimension-0 boundary, which
// is exactly where a shape checker is supposed to earn its keep.
func TestAnEmptyLiteralIsAList(t *testing.T) {
	for _, src := range []string{
		"print(sum([]))",
		"print(mean([]))",
		"print(exp([]))",
		"print(shape([]))",
	} {
		noFalseNegative(t, src)
	}
}

// A tensor builtin still takes its shape as a list of dimensions, and its axis
// as a number. Reporting those would be the false positive that makes a checker
// worth turning off, and it did: std/nn.tw and five other files tripped it.
func TestTensorBuiltinsStillTakeListShapes(t *testing.T) {
	for _, src := range []string{
		"print(reshape(zeros(2, 3), list(3, 2)))",
		"print(broadcast_to(zeros(1, 3), list(2, 3)))",
		"print(reshape(zeros(2, 3), 3, 2))",
		"print(sum(zeros(2, 3), 1))",
	} {
		noFalsePositive(t, src)
	}
}

// A scalar has no axes. Each axis check guarded on rank > 0 before validating,
// so the rank-0 case was the one branch that fell off the end.
func TestAnAxisOnAScalarIsReported(t *testing.T) {
	wantOne(t, "let r = sum(1.0, 0)", "a scalar has no axes")
	wantOne(t, "let r = mean(2.0, 1)", "a scalar has no axes")
	wantNone(t, "let r = sum(3.0)")
}

// Reading a field of something that cannot have one.
func TestAFieldOfANonRecordIsReported(t *testing.T) {
	wantOne(t, "mode systems\nfn f() -> I64 {\n  let x: I64 = 5\n  x.foo\n}", `cannot read field "foo" of I64`)
	wantOne(t, "mode systems\nfn f(s: Str) -> Str = s.len", `cannot read field "len" of Str`)
}

// `a.push(v)` is uniform call syntax for `push(a, v)` and works on any value,
// so a field in callee position is a function name and not a field at all.
// Reporting it broke warp's suite, which is how it was caught.
func TestUniformCallSyntaxIsNotAFieldRead(t *testing.T) {
	noFalsePositive(t, "mode systems\nfn main() {\n  let a: Arr[F64] = arr_new()\n  a.push(1.5)\n  print(len(a))\n}")
}

// `slice` is a Str slice and only a Str. The checker typed an Arr argument as
// an Arr result, which propagated a confident wrong type, so the definite
// mismatch never surfaced downstream either.
func TestSliceIsTypedAsTheRuntimeImplementsIt(t *testing.T) {
	wantOne(t, "mode systems\nfn f(a: Arr[Str]) -> Arr[Str] = slice(a, 0, 1)",
		"returns Str but its signature declares Arr[Str]")
	wantNone(t, "mode systems\nfn f(s: Str) -> Str = slice(s, 0, 1)")
}
