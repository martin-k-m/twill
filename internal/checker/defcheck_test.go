package checker_test

import "testing"

// These functions are never called; the errors are caught at definition.

func TestUncalledMatmulMismatch(t *testing.T) {
	wantOne(t, "fn broken(A: [2, 3]) = A @ [1.0, 2.0]", "inner")
}

func TestUncalledReturnMismatch(t *testing.T) {
	wantOne(t, "fn wrong(A: [2, 3]) -> [2] { sum(A, 0) }", "declares")
}

func TestUncalledElementwiseMismatch(t *testing.T) {
	wantOne(t, "fn bad(x: [2]) = x + [1.0, 2.0, 3.0]", "broadcast")
}

func TestUncalledFieldTypo(t *testing.T) {
	wantOne(t, "type M = { w: [2] }\nfn f(m: M) = sum(m.wrong)", "no field")
}

func TestUnannotatedFunctionNoFalsePositive(t *testing.T) {
	// Without annotations the parameter shapes are unknown, so a definition
	// that would only fail for some shapes is not flagged.
	wantNone(t, "fn dense(w, b, x) = w @ x + b")
	wantNone(t, "fn f(x) = x @ x")
}

func TestGenericShapeVarFunctionNoFalsePositive(t *testing.T) {
	// Shape variables are unknown sizes at definition, so a valid generic
	// function is not flagged.
	wantNone(t, "fn mm(A: [n, k], B: [k, m]) -> [n, m] { A @ B }")
}
