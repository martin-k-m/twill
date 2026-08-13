package interp_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/martin-k-m/twill/internal/interp"
	"github.com/martin-k-m/twill/internal/value"
)

// The self-hosted compiler is twill written in twill (src/main.tw and the
// modules it imports). Running it on the Go bootstrap and asking it to check a
// file exercises the whole front end -- lexer, parser and checker, all in twill
// -- end to end. Its exit code is the milestone under guard: 0 for a clean file,
// 1 for one with a shape error, matching what the Go `check` command returns.
func runSelfHostedCheck(t *testing.T, source string) int {
	t.Helper()
	dir := t.TempDir()
	target := filepath.Join(dir, "input.tw")
	if err := os.WriteFile(target, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	ip := interp.New(func(string) {})
	result, ranMain, err := ip.RunFileMain(filepath.Join("..", "..", "src", "main.tw"),
		[]string{"twill", "check", target})
	if err != nil {
		t.Fatalf("self-hosted CLI errored: %v", err)
	}
	if !ranMain {
		t.Fatal("self-hosted main did not run")
	}
	n, ok := value.AsNumber(result)
	if !ok {
		t.Fatalf("self-hosted main returned a non-number: %v", result)
	}
	return int(n)
}

// runBothWays evaluates a program on the Go interpreter directly and on the
// self-hosted evaluator (src/main.tw run) and returns the printed output of
// each. The self-hosted evaluator is meant to match the bootstrap byte for
// byte, so a builtin added to one is not done until it agrees on the other.
func runBothWays(t *testing.T, source string) (goOut, selfOut string) {
	t.Helper()
	var goBuf strings.Builder
	goIP := interp.New(func(s string) { goBuf.WriteString(s) })
	if _, err := goIP.Run(source); err != nil {
		t.Fatalf("Go interpreter errored: %v", err)
	}

	dir := t.TempDir()
	target := filepath.Join(dir, "input.tw")
	if err := os.WriteFile(target, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	var selfBuf strings.Builder
	selfIP := interp.New(func(s string) { selfBuf.WriteString(s) })
	if _, ranMain, err := selfIP.RunFileMain(filepath.Join("..", "..", "src", "main.tw"),
		[]string{"twill", "run", target}); err != nil {
		t.Fatalf("self-hosted CLI errored: %v", err)
	} else if !ranMain {
		t.Fatal("self-hosted main did not run")
	}
	return goBuf.String(), selfBuf.String()
}

// A list of strings sorts by the bytewise order docs/language-guide.md pins
// (NEEDS-23), on both the Go bootstrap and the self-hosted evaluator, and the
// two must agree. Uppercase sorts before lowercase (byte value), a shorter
// prefix sorts first, and the optional descending flag reverses the result.
func TestSelfHostedSortsAStringListLikeTheBootstrap(t *testing.T) {
	for _, src := range []string{
		`print(sort(["banana", "apple", "cherry", "Apple"]))`,
		`print(sort(["b", "a", "c"], true))`,
		`print(sort(["ab", "a", "abc"]))`,
		`print(sort(list("ab", "a", "abc"), true))`,
	} {
		goOut, selfOut := runBothWays(t, src+"\n")
		if goOut != selfOut {
			t.Fatalf("sort output diverged for %q:\n  go:   %q\n  self: %q", src, goOut, selfOut)
		}
	}
}

// linspace and arange are new construction builtins; the self-hosted evaluator
// must produce the same tensors as the bootstrap.
func TestSelfHostedConstructionBuiltinsMatch(t *testing.T) {
	for _, src := range []string{
		`print(linspace(0.0, 1.0, 5))`,
		`print(linspace(-2.0, 2.0, 9))`,
		`print(linspace(3.0, 3.0, 1))`,
		`print(arange(0.0, 2.0, 0.5))`,
		`print(arange(1.0, 10.0, 2.0))`,
		`print(arange(5.0, 5.0, 1.0))`,
	} {
		goOut, selfOut := runBothWays(t, src+"\n")
		if goOut != selfOut {
			t.Fatalf("output diverged for %q:\n  go:   %q\n  self: %q", src, goOut, selfOut)
		}
	}
}

func TestSelfHostedCheckCleanFile(t *testing.T) {
	if code := runSelfHostedCheck(t, "mode systems\nfn add(a: I64, b: I64) -> I64 = a + b\n"); code != 0 {
		t.Fatalf("check of a clean file exited %d, want 0", code)
	}
}

func TestSelfHostedCheckShapeError(t *testing.T) {
	src := "let a = [1.0, 2.0]\nlet b = [1.0, 2.0, 3.0]\nlet c = a + b\n"
	if code := runSelfHostedCheck(t, src); code != 1 {
		t.Fatalf("check of a mismatched file exited %d, want 1", code)
	}
}

// A shape error that hinges on the value of a numeric literal -- an axis out of
// range -- exercises the number parser end to end: the self-hosted lexer reads
// the token "7", the parser turns it into the value 7, and the checker compares
// it to the rank. Before str_to_f64 the literal parsed to garbage and this
// error was silently missed, so this guards that numbers reach the checker
// intact.
func TestSelfHostedCheckCatchesBadAxis(t *testing.T) {
	if code := runSelfHostedCheck(t, "let x = zeros(2, 3)\nlet y = sum(x, 7)\n"); code != 1 {
		t.Fatalf("check of an out-of-range axis exited %d, want 1", code)
	}
}

// topk/argtopk shorten an axis to k, so a k past the axis length is a runtime
// error the self-hosted checker must catch too -- and argtopk was unreachable
// until its name was added to the front end, so this also guards that.
func TestSelfHostedCheckCatchesTopKBounds(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = topk(zeros(3), 5)\n"); code != 1 {
		t.Fatalf("check of topk k past the axis exited %d, want 1", code)
	}
	if code := runSelfHostedCheck(t, "let y = argtopk(zeros(3), 5)\n"); code != 1 {
		t.Fatalf("check of argtopk k past the axis exited %d, want 1", code)
	}
	if code := runSelfHostedCheck(t, "let y = topk(zeros(3), 2)\n"); code != 0 {
		t.Fatalf("check of a valid topk exited %d, want 0", code)
	}
}

// split was unreachable until its name was added to the front end; both
// checkers must now accept a valid split rather than call it an unknown name.
func TestSelfHostedCheckAcceptsSplit(t *testing.T) {
	if code := runSelfHostedCheck(t, "let x = zeros(6)\nlet p = split(x, 2, 0)\n"); code != 0 {
		t.Fatalf("check of a valid split exited %d, want 0", code)
	}
	// The same bounds mismatches the runtime raises are caught statically, and the
	// self-hosted checker must agree: sizes that do not sum, and a bad divisor.
	if code := runSelfHostedCheck(t, "let x = zeros(6)\nlet p = split(x, [2, 2], 0)\n"); code != 1 {
		t.Fatalf("check of a split with sizes that do not sum exited %d, want 1", code)
	}
	if code := runSelfHostedCheck(t, "let x = zeros(7)\nlet p = split(x, 3, 0)\n"); code != 1 {
		t.Fatalf("check of a split count that does not divide exited %d, want 1", code)
	}
}

// The fixed-arity table catches the wrong argument count for any builtin the
// runtime registers with a set arity; the self-hosted checker owns the same
// table and must agree. A same-named user function keeps its own arity, and a
// variadic builtin is not constrained.
func TestSelfHostedCheckCatchesFixedArity(t *testing.T) {
	if code := runSelfHostedCheck(t, "let x = zeros(3, 8, 8)\nlet y = maxpool2d(x)\n"); code != 1 {
		t.Fatalf("check of maxpool2d with one argument exited %d, want 1", code)
	}
	if code := runSelfHostedCheck(t, "let y = clip(zeros(3), 0.0)\n"); code != 1 {
		t.Fatalf("check of clip with two arguments exited %d, want 1", code)
	}
	if code := runSelfHostedCheck(t, "let x = zeros(3, 8, 8)\nlet y = maxpool2d(x, 2)\n"); code != 0 {
		t.Fatalf("check of a valid maxpool2d exited %d, want 0", code)
	}
	// A variadic builtin (transpose takes an optional permutation) is not fixed.
	if code := runSelfHostedCheck(t, "let y = transpose(zeros(2, 3))\n"); code != 0 {
		t.Fatalf("check of a variadic transpose exited %d, want 0", code)
	}
}

// dtype(t) names a tensor's element type. Every tensor is f64 until a cast or a
// dtype-carrying constructor exists, so both interpreters must return "f64" and
// agree; this also guards that the self-hosted builtin is reachable (it was dead
// code, missing from the INSPECT dispatch list) and that Go recognises the name
// (it rejected it as unknown before dtype landed in the Go bootstrap).
func TestSelfHostedDtypeBuiltinMatches(t *testing.T) {
	goOut, selfOut := runBothWays(t, "let x = zeros(2, 3)\nprint(dtype(x))\nprint(dtype(sum(x)))\n")
	if goOut != selfOut {
		t.Fatalf("dtype output differs: Go %q vs self-hosted %q", goOut, selfOut)
	}
	if goOut != "f64f64" {
		t.Fatalf("dtype output = %q, want f64 twice", goOut)
	}
}

// An elementwise op produces the promoted dtype and rounds each element to it;
// both interpreters must agree. A comparison yields the promoted dtype, not bool
// (Go's compareOp returns a plain 0/1 tensor), so it is tested here too.
func TestSelfHostedElementwisePromotionMatches(t *testing.T) {
	src := "print(fill(0.1, 2, bf16) + ones(2, bf16))\n" +
		"print(ones(2, i8) + ones(2, i8))\n" +
		"print(greater(fill(1.0, 2, bf16), fill(2.0, 2, bf16)))\n" +
		"print(zeros(2) + ones(2))\n"
	goOut, selfOut := runBothWays(t, src)
	if goOut != selfOut {
		t.Fatalf("elementwise dtype differs:\n Go  %q\n self %q", goOut, selfOut)
	}
}

// A unary op keeps a float operand's dtype and rounds to it; both interpreters
// must agree, and f64 is left untouched.
func TestSelfHostedUnaryPromotionMatches(t *testing.T) {
	src := "print(relu(fill(0.5, 2, bf16)))\nprint(exp(fill(1.0, 2, bf16)))\n" +
		"print(square(fill(0.1, 2, bf16)))\nprint(relu(zeros(2)))\n"
	goOut, selfOut := runBothWays(t, src)
	if goOut != selfOut {
		t.Fatalf("unary dtype differs:\n Go  %q\n self %q", goOut, selfOut)
	}
}

// An operation whose result is an index produces i32 (docs/dtypes.md), so
// argmax/argsort/argtopk print with a dtype=i32 tag. This closes a long-standing
// divergence: the self-hosted evaluator always tagged them and the Go bootstrap,
// with no dtype, did not.
func TestSelfHostedIndexOpsAreI32(t *testing.T) {
	src := "print(argmax(tensor([[3.0, 1.0], [2.0, 5.0]]), 1))\n" +
		"print(argsort(tensor([3.0, 1.0, 4.0])))\n" +
		"print(argtopk(tensor([3.0, 1.0, 4.0, 1.5]), 2))\n"
	goOut, selfOut := runBothWays(t, src)
	if goOut != selfOut {
		t.Fatalf("index-op dtype differs:\n Go  %q\n self %q", goOut, selfOut)
	}
	if !strings.Contains(goOut, "dtype=i32") {
		t.Fatalf("expected an i32 tag, got %q", goOut)
	}
}

// Printing a narrow tensor shows the dtype tag and the dtype's shortest decimal,
// and both interpreters must agree. Positive values only: the self-hosted cast
// has a NEEDS-2 sign bug, so negatives are not a valid parity reference.
func TestSelfHostedNarrowPrintMatches(t *testing.T) {
	src := "print(fill(0.1, 3, bf16))\nprint(fill(3.14159, 2, f16))\nprint(ones(2, i8))\nprint(zeros(2, 2))\n"
	goOut, selfOut := runBothWays(t, src)
	if goOut != selfOut {
		t.Fatalf("narrow print differs:\n Go  %q\n self %q", goOut, selfOut)
	}
}

// A dtype-carrying constructor tags its tensor, and dtype() reads the tag back;
// both interpreters must agree on the name for every maker and both arities.
func TestSelfHostedConstructorDtypeMatches(t *testing.T) {
	src := "print(dtype(zeros(2, 3, bf16)))\nprint(dtype(eye(3, f32)))\n" +
		"print(dtype(scalar(1.5, i8)))\nprint(dtype(ones(4)))\n"
	goOut, selfOut := runBothWays(t, src)
	if goOut != selfOut {
		t.Fatalf("constructor dtype differs: Go %q vs self-hosted %q", goOut, selfOut)
	}
	if goOut != "bf16f32i8f64" {
		t.Fatalf("constructor dtype output = %q", goOut)
	}
}

// quantize packs a 2-D weight; the self-hosted checker owns the same rank rule
// even though its evaluator leaves quantize unimplemented (it types the result
// Unknown), so the check must still fire and a valid weight must pass.
func TestSelfHostedCheckCatchesQuantizeRank(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = quantize(zeros(5))\n"); code != 1 {
		t.Fatalf("check of quantize on a rank-1 weight exited %d, want 1", code)
	}
	if code := runSelfHostedCheck(t, "let y = quantize(zeros(4, 8))\n"); code != 0 {
		t.Fatalf("check of quantize on a 2-D weight exited %d, want 0", code)
	}
}

// A constant gather index past the first dimension is the runtime's
// out-of-range error; the self-hosted checker must fold it just as the Go one.
func TestSelfHostedCheckCatchesGatherIndex(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = gather(zeros(3, 2), [7])\n"); code != 1 {
		t.Fatalf("check of a gather index out of range exited %d, want 1", code)
	}
	if code := runSelfHostedCheck(t, "let y = gather(zeros(3, 2), [2, 0])\n"); code != 0 {
		t.Fatalf("check of an in-range gather exited %d, want 0", code)
	}
}

// An empty list handed to concat has nothing to join; the self-hosted checker
// must reject it just as the Go one does, keying on the literal.
func TestSelfHostedCheckCatchesEmptyConcat(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = concat([], 0)\n"); code != 1 {
		t.Fatalf("check of an empty concat exited %d, want 1", code)
	}
	if code := runSelfHostedCheck(t, "let y = concat([zeros(2), ones(3)], 0)\n"); code != 0 {
		t.Fatalf("check of a valid concat exited %d, want 0", code)
	}
}

// Loop-control checking (docs/language-guide.md, break/continue) is one of the
// non-shape diagnostics the checker owns, so the self-hosted checker must agree
// with the Go one on it: a break outside any loop is an error, a break inside a
// loop is clean, and a break in a function nested in a loop is an error because
// it cannot cross the function boundary.
func TestSelfHostedCheckCatchesBreakOutsideLoop(t *testing.T) {
	if code := runSelfHostedCheck(t, "let x = 1\nbreak\n"); code != 1 {
		t.Fatalf("check of a break outside a loop exited %d, want 1", code)
	}
}

func TestSelfHostedCheckAllowsBreakInsideLoop(t *testing.T) {
	if code := runSelfHostedCheck(t, "for i in range(3) {\n  if i == 1 { break }\n}\n"); code != 0 {
		t.Fatalf("check of a break inside a loop exited %d, want 0", code)
	}
}

func TestSelfHostedCheckCatchesBreakAcrossFunctionBoundary(t *testing.T) {
	src := "for i in range(3) {\n  let f = fn() { break }\n  f()\n}\n"
	if code := runSelfHostedCheck(t, src); code != 1 {
		t.Fatalf("check of a break across a function boundary exited %d, want 1", code)
	}
}

// transpose is the axis-taking builtin that used to stay silent on an
// out-of-range axis (NEEDS-50). The self-hosted checker must report it exactly
// as the Go one does, so a permutation naming a nonexistent axis fails the check
// in both.
func TestSelfHostedCheckCatchesBadTransposeAxis(t *testing.T) {
	if code := runSelfHostedCheck(t, "let x = zeros(2, 3)\nlet y = transpose(x, 0, 5)\n"); code != 1 {
		t.Fatalf("check of an out-of-range transpose axis exited %d, want 1", code)
	}
}

func TestSelfHostedCheckAllowsValidTranspose(t *testing.T) {
	if code := runSelfHostedCheck(t, "let x = zeros(2, 3)\nlet y = transpose(x, 1, 0)\n"); code != 0 {
		t.Fatalf("check of a valid transpose exited %d, want 0", code)
	}
}

// `@` has no batched form: the self-hosted checker rejects a rank-3 operand
// exactly as the Go one does, and passes the 2-D form.
func TestSelfHostedCheckRejectsRankThreeMatmul(t *testing.T) {
	if code := runSelfHostedCheck(t, "let a = zeros(5, 2, 3)\nlet b = zeros(5, 3, 4)\nlet c = a @ b\n"); code != 1 {
		t.Fatalf("check of a rank-3 matmul exited %d, want 1", code)
	}
}

func TestSelfHostedCheckAllowsTwoDMatmul(t *testing.T) {
	if code := runSelfHostedCheck(t, "let c = zeros(2, 3) @ zeros(3, 4)\n"); code != 0 {
		t.Fatalf("check of a 2-D matmul exited %d, want 0", code)
	}
}

// softmax's out-of-range axis is a diagnostic in the self-hosted checker too,
// and a valid axis is clean.
func TestSelfHostedCheckRejectsBadSoftmaxAxis(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = softmax(zeros(2, 3), 5)\n"); code != 1 {
		t.Fatalf("check of a bad softmax axis exited %d, want 1", code)
	}
}

func TestSelfHostedCheckAllowsGoodSoftmaxAxis(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = softmax(zeros(2, 3), 1)\n"); code != 0 {
		t.Fatalf("check of a valid softmax axis exited %d, want 0", code)
	}
}

// conv2d's three shape contracts are enforced by the self-hosted checker too.
func TestSelfHostedCheckRejectsBadConv2d(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = conv2d(zeros(3, 8, 8), zeros(4, 5, 3, 3))\n"); code != 1 {
		t.Fatalf("check of a conv2d channel mismatch exited %d, want 1", code)
	}
}

func TestSelfHostedCheckAllowsValidConv2d(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = conv2d(zeros(3, 8, 8), zeros(4, 3, 3, 3))\n"); code != 0 {
		t.Fatalf("check of a valid conv2d exited %d, want 0", code)
	}
}

// The shape-preserving axis ops (flip/cumsum/sort/roll) validate their axis in
// the self-hosted checker too, including roll's shift-first argument order.
func TestSelfHostedCheckRejectsBadRollAxis(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = roll(zeros(2, 3), 1, 5)\n"); code != 1 {
		t.Fatalf("check of a bad roll axis exited %d, want 1", code)
	}
}

func TestSelfHostedCheckAllowsValidFlip(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = flip(zeros(2, 3), 1)\n"); code != 0 {
		t.Fatalf("check of a valid flip exited %d, want 0", code)
	}
}

// broadcast_to compatibility and the list(...) shape form are checked by the
// self-hosted checker too.
func TestSelfHostedCheckRejectsBadBroadcast(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = broadcast_to(zeros(4), list(2, 3))\n"); code != 1 {
		t.Fatalf("check of an incompatible broadcast exited %d, want 1", code)
	}
}

func TestSelfHostedCheckReadsListShapeForReshape(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = reshape(zeros(6), list(2, 4))\n"); code != 1 {
		t.Fatalf("check of a bad reshape via list(...) exited %d, want 1", code)
	}
}

func TestSelfHostedCheckAllowsValidBroadcast(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = broadcast_to(zeros(3), list(2, 3))\n"); code != 0 {
		t.Fatalf("check of a valid broadcast exited %d, want 0", code)
	}
}

// A constant out-of-range index is caught by the self-hosted checker too.
func TestSelfHostedCheckRejectsOutOfRangeIndex(t *testing.T) {
	if code := runSelfHostedCheck(t, "let y = zeros(3)[3]\n"); code != 1 {
		t.Fatalf("check of an out-of-range index exited %d, want 1", code)
	}
}

func TestSelfHostedCheckRejectsScalarMatmul(t *testing.T) {
	if code := runSelfHostedCheck(t, "let c = scalar(2.0) @ zeros(3)\n"); code != 1 {
		t.Fatalf("check of a scalar matmul exited %d, want 1", code)
	}
}
