package interp_test

import (
	"math"
	"testing"

	"github.com/martin-k-m/raster/internal/interp"
	"github.com/martin-k-m/raster/internal/tensor"
)

func hessResult(t *testing.T, src string) *tensor.Tensor {
	t.Helper()
	v, _ := run(t, src)
	ht, ok := v.(*tensor.Tensor)
	if !ok {
		t.Fatalf("expected a tensor, got %T", v)
	}
	return ht
}

func checkHessian(t *testing.T, got *tensor.Tensor, n int, want []float64) {
	t.Helper()
	if len(got.Shape) != 2 || got.Shape[0] != n || got.Shape[1] != n {
		t.Fatalf("shape = %v, want [%d %d]", got.Shape, n, n)
	}
	for i := range want {
		if math.Abs(got.Data[i]-want[i]) > 1e-9 {
			t.Fatalf("hessian = %v, want %v", got.Data, want)
		}
	}
}

// f(x) = sum(x*x): Hessian is 2*I.
func TestHessianQuadratic(t *testing.T) {
	got := hessResult(t, `hessian(fn(x) = sum(x * x))([1.0, 2.0, 3.0])`)
	checkHessian(t, got, 3, []float64{
		2, 0, 0,
		0, 2, 0,
		0, 0, 2,
	})
}

// f(x) = sum(exp(x)): Hessian is diag(exp(x)).
func TestHessianExp(t *testing.T) {
	got := hessResult(t, `hessian(fn(x) = sum(exp(x)))([0.5, -1.0])`)
	checkHessian(t, got, 2, []float64{
		math.Exp(0.5), 0,
		0, math.Exp(-1.0),
	})
}

// A quadratic form f(x) = xᵀ A x has Hessian A + Aᵀ (constant), exercising the
// off-diagonal (cross-derivative) path via matmul.
func TestHessianQuadraticForm(t *testing.T) {
	// A = [[1,2],[3,4]]  ->  H = [[2,5],[5,8]]
	src := `
let A = [[1.0, 2.0], [3.0, 4.0]]
hessian(fn(x) = sum((A @ x) * x))([5.0, -2.0])`
	got := hessResult(t, src)
	checkHessian(t, got, 2, []float64{
		2, 5,
		5, 8,
	})
}

// The Hessian must flow through slicing (a linear structural op): for
// f(x) = sum(x[0:2] * x[2:4]) = x0*x2 + x1*x3, the only nonzero second partials
// are the cross terms d²f/dx0dx2 = d²f/dx1dx3 = 1.
func TestHessianThroughSlicing(t *testing.T) {
	got := hessResult(t, `hessian(fn(x) = sum(x[0:2] * x[2:4]))([1.0, 2.0, 3.0, 4.0])`)
	checkHessian(t, got, 4, []float64{
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 1, 0, 0,
	})
}

// A function using an op without forward-mode support must error, never return
// a silently-wrong Hessian.
func TestHessianUnsupportedOpErrors(t *testing.T) {
	ip := interp.New(func(string) {})
	if _, err := ip.Run(`hessian(fn(x) = max(x) + sum(x))([1.0, 2.0])`); err == nil {
		t.Fatal("expected an error for an op without forward-mode support")
	}
}

// floor is piecewise constant, so its second derivative is zero everywhere it
// is defined. It also returns an untracked tensor, which detaches the input
// from the graph, and hessian used to panic on that instead of answering.
// Zeros keeps it consistent with grad, which already returns zeros here.
func TestHessianDetachedInput(t *testing.T) {
	got := hessResult(t, `hessian(fn(x) = sum(floor(x)))([1.5, 2.5])`)
	checkHessian(t, got, 2, []float64{
		0, 0,
		0, 0,
	})
}

// A function that ignores its argument entirely reaches the same code path.
func TestHessianConstantExpression(t *testing.T) {
	got := hessResult(t, `hessian(fn(x) = 42.0)([1.0, 2.0, 3.0])`)
	checkHessian(t, got, 3, []float64{
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
	})
}
