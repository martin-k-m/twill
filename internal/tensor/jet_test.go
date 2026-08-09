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

// Sort is a permutation, so its jet must carry each tangent with its value.
// sum(square(sort(x))) has the same Hessian as sum(square(x)), 2I, but only if
// the sort jvp permutes the tangents rather than dropping or misplacing them.
// Distinct inputs keep the finite-difference perturbation away from a tie.
func TestHessianFDSort(t *testing.T) {
	checkHessianFD(t, "sort", []float64{3, 1, 4, 1.5, 2}, []int{5}, func(x *Tensor) *Tensor {
		s, _ := SortAxis(x, 0, false)
		return Sum(Square(s))
	})
}

// TopK keeps the k largest, so the kept elements carry curvature 2 and the
// dropped ones zero. The jet must carry tangents only through what survives.
func TestHessianFDTopK(t *testing.T) {
	checkHessianFD(t, "topk", []float64{3, 1, 4, 1.5, 2}, []int{5}, func(x *Tensor) *Tensor {
		s, _ := TopKAxis(x, 3, 0, true)
		return Sum(Square(s))
	})
}

// Softmax has a genuine second derivative, not a permutation, so this exercises
// the jvp's dd term. Two functions: the sum of squared probabilities, and a
// weighted log-loss over them, which are the two shapes a classifier hessian
// takes. Distinct-enough logits keep the jacobian well away from degeneracy.
func TestHessianFDSoftmax(t *testing.T) {
	checkHessianFD(t, "softmax-sq", []float64{0.5, -1.0, 2.0, 0.3}, []int{4}, func(x *Tensor) *Tensor {
		s, _ := Softmax(x, 0)
		return Sum(Square(s))
	})
	w := New([]float64{0.1, 0.3, 0.2, 0.25, 0.15}, []int{5})
	checkHessianFD(t, "softmax-logloss", []float64{0.2, 1.1, -0.7, 0.9, 0.0}, []int{5}, func(x *Tensor) *Tensor {
		s, _ := Softmax(x, 0)
		return Sum(mul(w, Log(s)))
	})
}

// LogSumExp reduces to a scalar and its gradient is softmax, so its second
// order is softmax's first order. Two functions: the square of the reduction,
// and lse(x) - x_target, which is the cross-entropy of a one-hot label and the
// most common place a hessian meets logsumexp.
func TestHessianFDLogSumExp(t *testing.T) {
	checkHessianFD(t, "lse-sq", []float64{0.5, -1.0, 2.0, 0.3}, []int{4}, func(x *Tensor) *Tensor {
		l, _ := LogSumExp(x, 0)
		return Square(l)
	})
	checkHessianFD(t, "lse-xent", []float64{0.2, 1.1, -0.7, 0.9, 0.0}, []int{5}, func(x *Tensor) *Tensor {
		l, _ := LogSumExp(x, 0)
		pick, _ := IndexAxis0(x, 2)
		return sub(l, pick)
	})
}

// Median is a selection, so its jet gathers the tangent it selects. Odd length
// picks a single middle element; even length averages the two, which splits the
// curvature of square(median) across both. Distinct values keep the selection
// off a tie, where an unstable sort makes the picked index arbitrary.
func TestHessianFDMedian(t *testing.T) {
	checkHessianFD(t, "median-odd", []float64{3, 1, 4, 1.5, 2.7}, []int{5}, func(x *Tensor) *Tensor {
		m, _ := MedianAxis(x, 0)
		return Square(m)
	})
	checkHessianFD(t, "median-even", []float64{3, 1, 4, 1.5}, []int{4}, func(x *Tensor) *Tensor {
		m, _ := MedianAxis(x, 0)
		return Square(m)
	})
}

// MaxPool2D is a selection, so its jet gathers the tangent of each window's max.
// A [1,2,4] input with a 2x2 window has two windows, so sum(square(pool)) curves
// only on the two winners. Distinct values keep each argmax off a tie.
func TestHessianFDMaxPool(t *testing.T) {
	checkHessianFD(t, "maxpool", []float64{3, 1, 4, 1.5, 2, 5, 0.5, 2.7}, []int{1, 2, 4}, func(x *Tensor) *Tensor {
		m, _ := MaxPool2D(x, 2)
		return Sum(Square(m))
	})
}

// Prod has a real second derivative. The no-zero path is checked against a
// finite-difference hessian; the zero paths cannot be, because a finite
// perturbation moves a factor off zero, so they are checked directly against the
// exact derivatives of a product polynomial. prod([2,0,3]) along [1,1,1] is
// t(2+t)(3+t) = 6t + 5t^2 + ..., so d=6 and dd=10; prod([0,0,4]) is 4t^2 + ...,
// so d=0 and dd=8.
func TestHessianFDProd(t *testing.T) {
	checkHessianFD(t, "prod-sq", []float64{1.5, -2.0, 0.7, 1.2}, []int{4}, func(x *Tensor) *Tensor {
		p, _ := ProdAxis(x, 0)
		return Square(p)
	})
	SetRecordJets(true)
	defer SetRecordJets(false)
	zc := func(name string, xd, v []float64, wantD, wantDD float64) {
		leaf := Leaf(xd, []int{len(xd)})
		leaf.RequiresGrad = true
		out, _ := ProdAxis(leaf, 0)
		d, dd, err := directional(out, leaf, []*Tensor{leaf, out}, v)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if math.Abs(d[0]-wantD) > 1e-9 || math.Abs(dd[0]-wantDD) > 1e-9 {
			t.Errorf("%s: d=%v dd=%v want d=%v dd=%v", name, d[0], dd[0], wantD, wantDD)
		}
	}
	zc("one-zero", []float64{2, 0, 3}, []float64{1, 1, 1}, 6, 10)
	zc("two-zero", []float64{0, 0, 4}, []float64{1, 1, 1}, 0, 8)
	zc("three-zero", []float64{0, 0, 0}, []float64{1, 1, 1}, 0, 0)
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
