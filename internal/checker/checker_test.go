package checker_test

import (
	"strings"
	"testing"

	"github.com/martin-k-m/raster/internal/checker"
	"github.com/martin-k-m/raster/internal/parser"
)

func diagnostics(t *testing.T, src string) []checker.Diagnostic {
	t.Helper()
	prog, err := parser.Parse(src)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	return checker.Check(prog)
}

func wantOne(t *testing.T, src, substr string) {
	t.Helper()
	diags := diagnostics(t, src)
	if len(diags) == 0 {
		t.Fatalf("expected a diagnostic containing %q, got none", substr)
	}
	found := false
	for _, d := range diags {
		if strings.Contains(d.Msg, substr) {
			found = true
		}
	}
	if !found {
		t.Fatalf("diagnostics %v did not contain %q", diags, substr)
	}
}

func wantNone(t *testing.T, src string) {
	t.Helper()
	if diags := diagnostics(t, src); len(diags) != 0 {
		t.Fatalf("expected no diagnostics, got %v", diags)
	}
}

func TestCatchesMatmulMismatch(t *testing.T) {
	wantOne(t, "let A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\nlet x = [1.0, 2.0]\nlet y = A @ x", "inner")
}

func TestCatchesElementwiseMismatch(t *testing.T) {
	wantOne(t, "let z = [1.0, 2.0] + [1.0, 2.0, 3.0]", "shape mismatch")
}

func TestCatchesRaggedLiteral(t *testing.T) {
	wantOne(t, "let m = [[1.0, 2.0], [3.0]]", "ragged")
}

func TestCatchesAnnotationMismatch(t *testing.T) {
	// f expects a length-3 vector but gets a length-2 one.
	wantOne(t, "fn f(v: [3]) = sum(v)\nlet r = f([1.0, 2.0])", "expects")
}

func TestGoodProgramHasNoDiagnostics(t *testing.T) {
	wantNone(t, `
		let A = [[1.0, 2.0], [3.0, 4.0]]
		let x = [1.0, 1.0]
		let y = A @ x + [0.5, 0.5]
		let s = mean(y * y)`)
}

func TestDynamicCodeNoFalsePositive(t *testing.T) {
	// Shapes here flow through grad/loops and cannot be fully known; the
	// checker must stay quiet rather than guess.
	wantNone(t, `
		let w = [0.0, 0.0]
		fn loss(w) = mean(w * w)
		for step in range(10) {
			let g = grad(loss)(w)
			w = w - g * 0.1
		}`)
}

// tensor(...) used to return an unknown type, which meant the shape of every
// literal was thrown away at the door and the checker had nothing to check.
func TestTensorLiteralKeepsItsShape(t *testing.T) {
	wantOne(t, `let a = tensor([[1.0, 2.0], [3.0, 4.0]])
let b = tensor([[1.0, 2.0, 3.0]])
print(a @ b)`, "inner 2 != 1")
}

func TestATensorLiteralThatFitsIsNotReported(t *testing.T) {
	// The half that decides whether anybody leaves the checker on.
	src := `let a = tensor([[1.0, 2.0], [3.0, 4.0]])
let b = tensor([[1.0], [2.0]])
print(a @ b)`
	if diags := diagnostics(t, src); len(diags) != 0 {
		t.Fatalf("a valid multiply was reported: %v", diags)
	}
}

func TestAFlatLiteralIsOneDimensional(t *testing.T) {
	wantOne(t, `let v = tensor([1.0, 2.0, 3.0])
let m = zeros(2, 2)
print(m @ v)`, "inner 2 != 3")
}

func TestARaggedLiteralIsLeftToTheRuntime(t *testing.T) {
	// It is already an error, and inventing a shape for it here would report a
	// second, imaginary problem somewhere downstream instead of the real one.
	for _, d := range diagnostics(t, `let a = tensor([[1.0, 2.0], [3.0]])
print(a @ a)`) {
		if strings.Contains(d.Msg, "inner") {
			t.Errorf("a ragged literal produced an invented shape error: %v", d.Msg)
		}
	}
}

func TestReshapeThatChangesTheElementCount(t *testing.T) {
	// The second most common shape mistake after a bad matmul, and it used to
	// reach the runtime untouched.
	wantOne(t, "let x = zeros(2, 3)\nprint(reshape(x, 4, 2))", "changes the number of elements")
}

func TestReshapeThatFitsIsQuiet(t *testing.T) {
	if diags := diagnostics(t, "let x = zeros(2, 3)\nprint(reshape(x, 3, 2))"); len(diags) != 0 {
		t.Fatalf("a valid reshape was reported: %v", diags)
	}
}

func TestAnAxisThatDoesNotExistIsReported(t *testing.T) {
	// Both reduction paths already worked this out and then returned an unknown
	// type, which silenced everything downstream too.
	wantOne(t, "let x = zeros(2, 3)\nprint(sum(x, 7))", "axis out of range")
	wantOne(t, "let x = zeros(2, 3)\nprint(argmax(x, 5))", "axis out of range")
}

func TestANegativeAxisStillCountsFromTheEnd(t *testing.T) {
	for _, src := range []string{
		"let x = zeros(2, 3)\nprint(sum(x, -1))",
		"let x = zeros(2, 3)\nprint(mean(x, 0))",
	} {
		if diags := diagnostics(t, src); len(diags) != 0 {
			t.Errorf("%q was reported: %v", src, diags)
		}
	}
}

func TestAnUnknownNameIsReported(t *testing.T) {
	wantOne(t, "let x = 1.0\nprint(nope + x)", `unknown name "nope"`)
}

func TestAFunctionDeclaredLaterIsNotUnknown(t *testing.T) {
	// A file may call a function declared further down, and does at run time.
	// Walking strictly in order would report a name that is perfectly defined.
	src := `fn caller(x) = helper(x) * 2.0
fn helper(x) = x + 1.0
print(caller(3.0))`
	if diags := diagnostics(t, src); len(diags) != 0 {
		t.Fatalf("a forward reference was reported: %v", diags)
	}
}

func TestAnUnaliasedImportSilencesTheNameCheck(t *testing.T) {
	// It brings its names in unqualified and the checker does not read the
	// imported file, so it cannot know what those names are. Guessing would
	// report definitions that exist.
	src := "import \"std/nn\"\nlet x = 1.0\nprint(whatever_nn_defines + x)"
	if diags := diagnostics(t, src); len(diags) != 0 {
		t.Fatalf("a blind import still reported unknown names: %v", diags)
	}
}

func TestAnAliasedImportKeepsTheNameCheck(t *testing.T) {
	// Every borrowed name arrives with the alias on it, so an unqualified name
	// is still provably nothing.
	wantOne(t, "import \"std/nn\" as nn\nprint(nope)", `unknown name "nope"`)
}

func TestConcatReportsPiecesThatDoNotFit(t *testing.T) {
	wantOne(t, "let a = zeros(2, 3)\nlet b = zeros(4, 5)\nprint(concat([a, b], 0))",
		"shapes differ on axis 1")
}

func TestConcatShapeFlowsDownstream(t *testing.T) {
	// The second half of the win: concat used to return an unknown type, so a
	// whole pipeline built on one was unchecked from that point on.
	wantOne(t, `let a = zeros(2, 3)
let b = zeros(2, 3)
let c = concat([a, b], 0)
print(c @ zeros(9, 9))`, "[4, 3] @ [9, 9]")
}

func TestConcatOnAnotherAxisAddsUpThere(t *testing.T) {
	src := `let a = zeros(2, 3)
let b = zeros(2, 5)
print(concat([a, b], 1) @ zeros(8, 2))`
	if diags := diagnostics(t, src); len(diags) != 0 {
		t.Fatalf("a valid join was reported: %v", diags)
	}
}

func TestConcatWithAnAxisThatDoesNotExist(t *testing.T) {
	wantOne(t, "let a = zeros(2, 3)\nprint(concat([a, a], 9))", "axis out of range")
}

func TestConcatSaysNothingWhenAPieceIsUnknown(t *testing.T) {
	// Unknowable is not the same as wrong.
	src := "let a = zeros(2, 3)\nlet b = load(\"x.npy\")\nprint(concat([a, b], 0))"
	if diags := diagnostics(t, src); len(diags) != 0 {
		t.Fatalf("an unknowable concat was reported: %v", diags)
	}
}
