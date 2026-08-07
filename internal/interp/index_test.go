package interp_test

import (
	"testing"

	"github.com/martin-k-m/twill/internal/tensor"
)

// Element indexing x[i] is now differentiable: grad flows to the indexed
// components. f(x) = x0² + x1·x2  ->  ∇f = [2x0, x2, x1].
func TestGradThroughElementIndex(t *testing.T) {
	v, _ := run(t, `grad(fn(x) = x[0] * x[0] + x[1] * x[2])([3.0, 5.0, 7.0])`)
	g, ok := v.(*tensor.Tensor)
	if !ok {
		t.Fatalf("expected a tensor, got %T", v)
	}
	want := []float64{6, 7, 5}
	if len(g.Data) != 3 {
		t.Fatalf("gradient length %d, want 3", len(g.Data))
	}
	for i, w := range want {
		if g.Data[i] != w {
			t.Fatalf("grad = %v, want %v", g.Data, want)
		}
	}
}

// And the Hessian flows through element indexing too: f(x) = x0·x1 has
// Hessian [[0,1],[1,0]].
func TestHessianThroughElementIndex(t *testing.T) {
	got := hessResult(t, `hessian(fn(x) = x[0] * x[1])([2.0, 3.0])`)
	checkHessian(t, got, 2, []float64{
		0, 1,
		1, 0,
	})
}
