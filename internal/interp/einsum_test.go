package interp_test

import "testing"

func TestEinsumMatmulLang(t *testing.T) {
	// [[1,2],[3,4]] @ [[5],[6]] = [[17],[39]]
	if got := scalar(t, `einsum("ij,jk->ik", [[1.0,2.0],[3.0,4.0]], [[5.0],[6.0]])[1][0]`); got != 39 {
		t.Errorf("got %v, want 39", got)
	}
}

func TestEinsumReductionsLang(t *testing.T) {
	if got := scalar(t, `einsum("ij->", [[1.0,2.0],[3.0,4.0]])`); got != 10 {
		t.Errorf("full sum got %v", got)
	}
	if got := scalar(t, `einsum("ij->ji", [[1.0,2.0,3.0],[4.0,5.0,6.0]])[2][1]`); got != 6 {
		t.Errorf("transpose got %v", got)
	}
}

func TestEinsumDifferentiable(t *testing.T) {
	// bil(x) = x . W . [1,1] = 3*x0 + 7*x1 ; grad = [3, 7].
	src := `
		let W = [[1.0, 2.0], [3.0, 4.0]]
		fn bil(x) = einsum("i,ij,j->", x, W, [1.0, 1.0])
		grad(bil)([1.0, 1.0])[1]`
	if got := scalar(t, src); got != 7 {
		t.Errorf("grad got %v, want 7", got)
	}
}
