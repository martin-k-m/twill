package tensor

import (
	"math"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

func TestScalarAddMul(t *testing.T) {
	a, b := Scalar(3), Scalar(4)
	sum, _ := Add(a, b)
	prod, _ := Mul(a, b)
	if sum.Data[0] != 7 || prod.Data[0] != 12 {
		t.Fatalf("got sum=%v prod=%v", sum.Data[0], prod.Data[0])
	}
}

func TestGradSquare(t *testing.T) {
	x := Leaf([]float64{5}, []int{})
	y, _ := Mul(x, x)
	if err := y.Backward(); err != nil {
		t.Fatal(err)
	}
	if y.Data[0] != 25 || x.Grad[0] != 10 {
		t.Fatalf("y=%v dx=%v", y.Data[0], x.Grad[0])
	}
}

func TestChainRule(t *testing.T) {
	// f(x) = (x - 1) / x ; f'(x) = 1/x^2
	x := Leaf([]float64{4}, []int{})
	num, _ := Sub(x, Scalar(1))
	f, _ := Div(num, x)
	if err := f.Backward(); err != nil {
		t.Fatal(err)
	}
	if !approx(f.Data[0], 0.75) || !approx(x.Grad[0], 1.0/16) {
		t.Fatalf("f=%v dx=%v", f.Data[0], x.Grad[0])
	}
}

func TestVectorSumGrad(t *testing.T) {
	v := Leaf([]float64{1, 2, 3}, []int{3})
	s := Sum(v)
	if err := s.Backward(); err != nil {
		t.Fatal(err)
	}
	if s.Data[0] != 6 {
		t.Fatalf("sum=%v", s.Data[0])
	}
	for i, g := range v.Grad {
		if g != 1 {
			t.Fatalf("grad[%d]=%v", i, g)
		}
	}
}

func TestReluGrad(t *testing.T) {
	v := Leaf([]float64{-2, 0.5, 3}, []int{3})
	s := Sum(Relu(v))
	if err := s.Backward(); err != nil {
		t.Fatal(err)
	}
	want := []float64{0, 1, 1}
	for i := range want {
		if v.Grad[i] != want[i] {
			t.Fatalf("grad=%v want %v", v.Grad, want)
		}
	}
}

func TestMatMulGrad(t *testing.T) {
	// y = sum(A @ x); dy/dA = rows of x; dy/dx = column sums of A.
	A := Leaf([]float64{1, 2, 3, 4}, []int{2, 2})
	x := Leaf([]float64{5, 6}, []int{2})
	prod, _ := MatMul(A, x)
	y := Sum(prod)
	if err := y.Backward(); err != nil {
		t.Fatal(err)
	}
	if y.Data[0] != 56 {
		t.Fatalf("y=%v", y.Data[0])
	}
	wantA := []float64{5, 6, 5, 6}
	for i := range wantA {
		if A.Grad[i] != wantA[i] {
			t.Fatalf("dA=%v want %v", A.Grad, wantA)
		}
	}
	wantX := []float64{4, 6}
	for i := range wantX {
		if x.Grad[i] != wantX[i] {
			t.Fatalf("dx=%v want %v", x.Grad, wantX)
		}
	}
}

func TestExpTanhDeriv(t *testing.T) {
	x := Leaf([]float64{0.7}, []int{})
	y := Exp(x)
	if err := y.Backward(); err != nil {
		t.Fatal(err)
	}
	if !approx(x.Grad[0], math.Exp(0.7)) {
		t.Fatalf("d exp=%v", x.Grad[0])
	}
	z := Leaf([]float64{0.3}, []int{})
	tt := Tanh(z)
	if err := tt.Backward(); err != nil {
		t.Fatal(err)
	}
	if !approx(z.Grad[0], 1-math.Tanh(0.3)*math.Tanh(0.3)) {
		t.Fatalf("d tanh=%v", z.Grad[0])
	}
}

func TestGradAccumulatesOnReuse(t *testing.T) {
	// f(x) = x + x = 2x ; f'(x) = 2
	x := Leaf([]float64{9}, []int{})
	y, _ := Add(x, x)
	if err := y.Backward(); err != nil {
		t.Fatal(err)
	}
	if x.Grad[0] != 2 {
		t.Fatalf("dx=%v", x.Grad[0])
	}
}

func TestMatMulShapeMismatch(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := New([]float64{1, 2}, []int{2})
	if _, err := MatMul(a, b); err == nil {
		t.Fatal("expected a shape-mismatch error")
	}
}

// The scalar fast path skips the broadcast machinery, so these check it agrees
// with the path it skips.
func TestScalarFastPathMatchesTheGeneralOne(t *testing.T) {
	plain := Scalar(3)
	other := Scalar(4)
	sum, err := Add(plain, other)
	if err != nil {
		t.Fatal(err)
	}
	if !sum.IsScalar() || sum.Data[0] != 7 {
		t.Errorf("3 + 4 = %v (shape %v)", sum.Data, sum.Shape)
	}
}

func TestScalarGradientsStillFlow(t *testing.T) {
	// The gate: anything requiring a gradient must take the general path, which
	// is where the backward closure lives.
	x := Leaf([]float64{2}, []int{})
	sq, err := Mul(x, x)
	if err != nil {
		t.Fatal(err)
	}
	if err := sq.Backward(); err != nil {
		t.Fatal(err)
	}
	if x.Grad[0] != 4 {
		t.Errorf("d(x*x)/dx at 2 = %v, want 4", x.Grad[0])
	}
}
