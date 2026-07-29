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
