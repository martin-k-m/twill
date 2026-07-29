package interp_test

import (
	"testing"

	"github.com/martin-k-m/raster/internal/tensor"
)

func jacResult(t *testing.T, src string) *tensor.Tensor {
	t.Helper()
	v, _ := run(t, src)
	jt, ok := v.(*tensor.Tensor)
	if !ok {
		t.Fatalf("expected a tensor, got %T", v)
	}
	return jt
}

func checkMatrix(t *testing.T, got *tensor.Tensor, rows, cols int, want []float64) {
	t.Helper()
	if len(got.Shape) != 2 || got.Shape[0] != rows || got.Shape[1] != cols {
		t.Fatalf("shape = %v, want [%d %d]", got.Shape, rows, cols)
	}
	for i := range want {
		if got.Data[i] != want[i] {
			t.Fatalf("jacobian = %v, want %v", got.Data, want)
		}
	}
}

// The Jacobian of a linear map A@x is exactly A, at any point.
func TestJacobianLinear(t *testing.T) {
	src := `
let A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
jacobian(fn(x) = A @ x)([0.5, -1.0, 2.0])`
	got := jacResult(t, src)
	checkMatrix(t, got, 2, 3, []float64{1, 2, 3, 4, 5, 6})
}

// The Jacobian of x*x is diag(2x).
func TestJacobianElementwise(t *testing.T) {
	got := jacResult(t, `jacobian(fn(x) = x * x)([2.0, 3.0, 4.0])`)
	checkMatrix(t, got, 3, 3, []float64{
		4, 0, 0,
		0, 6, 0,
		0, 0, 8,
	})
}

// A nonlinear map with genuine cross-terms: f(x) = [x0*x1, x1*x2].
// J = [[x1, x0, 0], [0, x2, x1]] = [[3,2,0],[0,4,3]] at (2,3,4).
func TestJacobianCrossTerms(t *testing.T) {
	src := `
fn f(x) {
  let a = x[0:1] * x[1:2]
  let b = x[1:2] * x[2:3]
  concat([a, b], 0)
}
jacobian(f)([2.0, 3.0, 4.0])`
	got := jacResult(t, src)
	checkMatrix(t, got, 2, 3, []float64{
		3, 2, 0,
		0, 4, 3,
	})
}
