package tensor

import (
	"math"
	"testing"
)

// checkHessianFD compares the forward-mode (jet) Hessian against a
// central-difference numerical Hessian, as an independent correctness check.
func checkHessianFD(t *testing.T, name string, xdata []float64, shape []int, f func(*Tensor) *Tensor) {
	t.Helper()
	SetRecordJets(true)
	defer SetRecordJets(false)
	leaf := Leaf(xdata, shape)
	out := f(leaf)
	H, n, err := Hessian(out, leaf)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	eval := func(xd []float64) float64 {
		return f(New(append([]float64(nil), xd...), shape)).Data[0]
	}
	const h = 1e-3
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			pert := func(si, sj float64) float64 {
				y := append([]float64(nil), xdata...)
				y[i] += si * h
				y[j] += sj * h
				return eval(y)
			}
			num := (pert(1, 1) - pert(1, -1) - pert(-1, 1) + pert(-1, -1)) / (4 * h * h)
			if math.Abs(num-H[i*n+j]) > 1e-4 {
				t.Errorf("%s: H[%d,%d] jet=%v finite-diff=%v", name, i, j, H[i*n+j], num)
			}
		}
	}
}

func mul(a, b *Tensor) *Tensor { r, _ := Mul(a, b); return r }
func add(a, b *Tensor) *Tensor { r, _ := Add(a, b); return r }
func sub(a, b *Tensor) *Tensor { r, _ := Sub(a, b); return r }
func div(a, b *Tensor) *Tensor { r, _ := Div(a, b); return r }
func matmul(a, b *Tensor) *Tensor {
	r, _ := MatMul(a, b)
	return r
}

// Variance: exercises a scalar broadcast (x - mean(x)) inside the jet path.
func TestHessianFDVariance(t *testing.T) {
	checkHessianFD(t, "variance", []float64{1, -2, 3, 0.5}, []int{4}, func(x *Tensor) *Tensor {
		return Mean(Square(sub(x, Mean(x))))
	})
}

// Reciprocal and a rational: exercises Div's second derivatives.
func TestHessianFDDivision(t *testing.T) {
	ones := Filled([]int{3}, 1)
	checkHessianFD(t, "reciprocal", []float64{0.7, 1.5, 2.2}, []int{3}, func(x *Tensor) *Tensor {
		return Sum(div(ones, x))
	})
	two := Scalar(2)
	checkHessianFD(t, "rational", []float64{0.3, -0.8, 1.1}, []int{3}, func(x *Tensor) *Tensor {
		return Sum(div(x, add(Square(x), two)))
	})
}

// A general (non-scalar) broadcast: a [2,1] column added across a [2,3] matrix,
// then exp — exercises jetBinary's odometer path.
func TestHessianFDColumnBroadcast(t *testing.T) {
	col := New([]float64{0.5, -0.5}, []int{2, 1})
	checkHessianFD(t, "col-broadcast", []float64{0.1, 0.2, 0.3, -0.1, 0.0, 0.4}, []int{2, 3}, func(x *Tensor) *Tensor {
		return Sum(Exp(add(x, col)))
	})
}

// A mix of transcendentals, so every unary second-derivative rule is checked.
func TestHessianFDTranscendentals(t *testing.T) {
	checkHessianFD(t, "transcendentals", []float64{0.4, 0.9, 1.3}, []int{3}, func(x *Tensor) *Tensor {
		terms := add(Tanh(x), add(Sin(x), add(Sigmoid(x), add(Log(x), add(Sqrt(x), PowScalar(x, 1.5))))))
		return Sum(mul(x, terms))
	})
}

// Structural/linear ops (slice, reshape, transpose, concat, gather) must carry
// forward-mode tangents so a Hessian flows through them.
func TestHessianFDSliceConcat(t *testing.T) {
	// f(x) = sum(square( x[0:2] ++ x[2:4] reversed via gather )) with a product
	// coupling the halves, exercising slice, concat, and cross terms.
	checkHessianFD(t, "slice-concat", []float64{0.5, -1, 2, 0.3}, []int{4}, func(x *Tensor) *Tensor {
		lo, _ := SliceAxis0(x, 0, 2)
		hi, _ := SliceAxis0(x, 2, 4)
		cat, _ := Concat([]*Tensor{hi, lo}, 0) // [x2,x3,x0,x1]
		return Sum(mul(x, cat))                // sum x_i * cat_i -> cross terms
	})
}

func TestHessianFDReshapeTranspose(t *testing.T) {
	// f(x) = sum(square(Xᵀ)) with x reshaped to a matrix then transposed.
	checkHessianFD(t, "reshape-transpose", []float64{0.2, -0.4, 0.6, 0.1, 0.5, -0.3}, []int{6}, func(x *Tensor) *Tensor {
		m, _ := Reshape(x, []int{2, 3})
		mt, _ := TransposePerm(m, nil)
		return Sum(mul(mt, mt))
	})
}

func TestHessianFDGather(t *testing.T) {
	// f(x) = sum(square(gather(x, [2,0,2,1]))) — repeated index couples entries.
	checkHessianFD(t, "gather", []float64{0.4, -0.7, 1.2}, []int{3}, func(x *Tensor) *Tensor {
		g, _ := Gather(x, []int{2, 0, 2, 1})
		return Sum(Square(g))
	})
}

// Cumulative scans: second derivatives must flow through the fold too.
func TestHessianFDCumulative(t *testing.T) {
	// sum(square(cumsum(x))) couples every pair i <= j.
	checkHessianFD(t, "cumsum", []float64{0.5, -1, 2, 0.3}, []int{4}, func(x *Tensor) *Tensor {
		return Sum(Square(CumSum(x)))
	})
	// cumprod is where the product rule has to be carried to second order.
	checkHessianFD(t, "cumprod", []float64{1.2, -0.6, 1.5}, []int{3}, func(x *Tensor) *Tensor {
		return Sum(CumProd(x))
	})
	// Piecewise linear, so the running extreme only routes tangents through.
	checkHessianFD(t, "cummax", []float64{1, 3, 2, 5}, []int{4}, func(x *Tensor) *Tensor {
		return Sum(Square(CumMax(x)))
	})
	checkHessianFD(t, "cummin", []float64{5, 3, 4, 1}, []int{4}, func(x *Tensor) *Tensor {
		return Sum(Square(CumMin(x)))
	})
}

// A quadratic through matmul with a nonlinear head.
func TestHessianFDMatmulNonlinear(t *testing.T) {
	A := New([]float64{1, 2, 3, 0.5, 1.5, -1, 2, -0.5, 1}, []int{3, 3})
	checkHessianFD(t, "matmul-nonlinear", []float64{0.3, -0.2, 0.5}, []int{3}, func(x *Tensor) *Tensor {
		return Sum(Square(matmul(A, x)))
	})
}

// A leaf the output does not descend from. The forward-only builtins (floor,
// ceil, round, the comparisons) return a fresh tensor with no parent edges, so
// walking back from the output never reaches the input. Hessian used to hand
// that leaf to directional anyway, which seeds a tangent into jet state the
// init loop had never allocated, and the copy dereferenced nil.
func TestHessianLeafNotInGraph(t *testing.T) {
	SetRecordJets(true)
	defer SetRecordJets(false)
	leaf := Leaf([]float64{1.5, 2.5}, []int{2})
	// Stands in for floor(leaf): the same values, none of the edges.
	detached := New([]float64{1, 2}, []int{2})
	H, n, err := Hessian(Sum(detached), leaf)
	if err != nil {
		t.Fatalf("Hessian: %v", err)
	}
	if n != 2 {
		t.Fatalf("n = %d, want 2", n)
	}
	for i, v := range H {
		if v != 0 {
			t.Fatalf("H[%d] = %v, want 0 (H = %v)", i, v, H)
		}
	}
}

// The same path reached the other way: a function that never mentions its
// argument. Its second derivative is zero, not an error.
func TestHessianConstantFunction(t *testing.T) {
	SetRecordJets(true)
	defer SetRecordJets(false)
	leaf := Leaf([]float64{3, -1, 0.5}, []int{3})
	H, n, err := Hessian(Sum(Filled([]int{3}, 7)), leaf)
	if err != nil {
		t.Fatalf("Hessian: %v", err)
	}
	if n != 3 || len(H) != 9 {
		t.Fatalf("n = %d, len(H) = %d, want 3 and 9", n, len(H))
	}
	for i, v := range H {
		if v != 0 {
			t.Fatalf("H[%d] = %v, want 0", i, v)
		}
	}
}
