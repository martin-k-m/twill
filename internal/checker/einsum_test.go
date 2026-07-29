package checker_test

import "testing"

func TestEinsumOutputShapeInferred(t *testing.T) {
	// einsum("ij,jk->ik") of [2,3] and [3,4] is [2,4]; @ with a length-2 vector
	// then has an inner-dim mismatch (4 != 2), which the checker catches.
	src := `
		let a = [[1.0,2.0,3.0],[4.0,5.0,6.0]]
		let b = [[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]]
		let c = einsum("ij,jk->ik", a, b)
		let y = c @ [1.0, 2.0]`
	wantOne(t, src, "inner")
}

func TestEinsumValidNoDiagnostics(t *testing.T) {
	src := `
		let a = [[1.0,2.0,3.0],[4.0,5.0,6.0]]
		let b = [[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]]
		let c = einsum("ij,jk->ik", a, b) + [0.0, 0.0, 0.0, 0.0]`
	wantNone(t, src)
}

func TestEinsumMalformedSpecCaught(t *testing.T) {
	// Spec names two operands but only one is passed.
	wantOne(t, `let y = einsum("ij,jk->ik", [[1.0,2.0]])`, "operands")
	// Rank mismatch: subscript "ij" needs rank 2 but the tensor is rank 1.
	wantOne(t, `let y = einsum("ij->i", [1.0, 2.0])`, "rank")
}
