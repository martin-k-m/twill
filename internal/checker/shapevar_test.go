package checker_test

import "testing"

func TestShapeVariablesConsistent(t *testing.T) {
	src := `
		fn mm(A: [n, k], B: [k, m]) -> [n, m] { A @ B }
		let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
		let b = [[1.0], [2.0], [3.0]]
		let c = mm(a, b)`
	wantNone(t, src)
}

func TestShapeVariableConflictCaught(t *testing.T) {
	// B's first axis must equal k (=3 from A) but is 2.
	src := `
		fn mm(A: [n, k], B: [k, m]) -> [n, m] { A @ B }
		let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
		let bad = mm(a, [[1.0, 2.0], [3.0, 4.0]])`
	wantOne(t, src, "shape variable")
}

func TestReturnAnnotationChecked(t *testing.T) {
	// Declares it returns [k, n] but the body returns [n, k].
	src := `
		fn wrong(A: [n, k]) -> [k, n] { A }
		let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
		let r = wrong(a)`
	wantOne(t, src, "signature declares")
}

func TestRankMismatchCaught(t *testing.T) {
	src := `
		fn f(A: [n, k]) -> [n] { sum(A, 1) }
		let v = [1.0, 2.0, 3.0]
		let r = f(v)`
	wantOne(t, src, "rank")
}

func TestConcreteAnnotationStillWorks(t *testing.T) {
	wantNone(t, "fn matvec(A: [3, 2], x: [2]) -> [3] { A @ x }")
	wantOne(t, "fn f(v: [3]) = sum(v)\nlet r = f([1.0, 2.0])", "axis")
}
