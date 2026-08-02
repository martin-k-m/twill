package tensor

import (
	"math"
	"testing"
)

// gradCheck compares the analytic gradient of build(x) (a scalar) against a
// central-difference numeric gradient.
func gradCheck(t *testing.T, name string, data []float64, shape []int, build func(*Tensor) *Tensor) {
	t.Helper()
	leaf := Leaf(data, shape)
	out := build(leaf)
	if !out.IsScalar() {
		t.Fatalf("%s: build must return a scalar, got shape %v", name, out.Shape)
	}
	if err := out.Backward(); err != nil {
		t.Fatalf("%s: backward: %v", name, err)
	}
	const eps = 1e-6
	for i := range data {
		orig := data[i]
		plus := append([]float64(nil), data...)
		plus[i] = orig + eps
		minus := append([]float64(nil), data...)
		minus[i] = orig - eps
		vp := build(New(plus, shape)).Data[0]
		vm := build(New(minus, shape)).Data[0]
		num := (vp - vm) / (2 * eps)
		ana := leaf.Grad[i]
		if math.Abs(num-ana) > 1e-4 {
			t.Errorf("%s: grad[%d] analytic=%v numeric=%v", name, i, ana, num)
		}
	}
}

func TestGradCheckBroadcastMul(t *testing.T) {
	// f(x) = sum(x * row) where row broadcasts over rows of x.
	row := New([]float64{2, 3}, []int{2})
	gradCheck(t, "broadcast-mul", []float64{1, 2, 3, 4}, []int{2, 2}, func(x *Tensor) *Tensor {
		p, _ := Mul(x, row)
		return Sum(p)
	})
}

func TestGradCheckBroadcastAddColumn(t *testing.T) {
	// Column vector [2,1] broadcasts across a [2,3] matrix.
	col := New([]float64{5, 7}, []int{2, 1})
	gradCheck(t, "broadcast-add-col", []float64{1, 2, 3, 4, 5, 6}, []int{2, 3}, func(x *Tensor) *Tensor {
		s, _ := Add(x, col)
		return Sum(s)
	})
}

func TestGradCheckSquareMean(t *testing.T) {
	gradCheck(t, "square-mean", []float64{-1, 0.5, 2, -3}, []int{4}, func(x *Tensor) *Tensor {
		return Mean(Square(x))
	})
}

func TestGradCheckMaximum(t *testing.T) {
	other := New([]float64{0, 1, 2, 3}, []int{4})
	gradCheck(t, "maximum", []float64{0.5, 0.5, 2.5, 2.5}, []int{4}, func(x *Tensor) *Tensor {
		m, _ := Maximum(x, other)
		return Sum(m)
	})
}

func TestGradCheckClip(t *testing.T) {
	gradCheck(t, "clip", []float64{-2, -0.5, 0.3, 1.5}, []int{4}, func(x *Tensor) *Tensor {
		return Sum(Clip(x, -1, 1))
	})
}

func TestGradCheckWhere(t *testing.T) {
	cond := New([]float64{1, 0, 1, 0}, []int{4})
	other := New([]float64{9, 9, 9, 9}, []int{4})
	gradCheck(t, "where", []float64{1, 2, 3, 4}, []int{4}, func(x *Tensor) *Tensor {
		w, _ := Where(cond, x, other)
		return Sum(w)
	})
}

func TestGradCheckSumAxis(t *testing.T) {
	gradCheck(t, "sum-axis", []float64{1, 2, 3, 4, 5, 6}, []int{2, 3}, func(x *Tensor) *Tensor {
		r, _ := SumAxis(x, 1)
		return Sum(Square(r))
	})
}

func TestGradCheckMeanAxis(t *testing.T) {
	gradCheck(t, "mean-axis", []float64{1, 2, 3, 4, 5, 6}, []int{2, 3}, func(x *Tensor) *Tensor {
		r, _ := MeanAxis(x, 0)
		return Sum(Square(r))
	})
}

func TestGradCheckMaxAxis(t *testing.T) {
	gradCheck(t, "max-axis", []float64{1, 5, 3, 2, 4, 6}, []int{2, 3}, func(x *Tensor) *Tensor {
		r, _ := MaxAxis(x, 1)
		return Sum(r)
	})
}

func TestGradCheckCumSum(t *testing.T) {
	// Weight the scan so each output carries a different gradient, which
	// catches a backward pass that reverses the accumulation the wrong way.
	w := New([]float64{0.5, -1, 2, 0.25}, []int{4})
	gradCheck(t, "cumsum", []float64{1, -2, 0.5, 3}, []int{4}, func(x *Tensor) *Tensor {
		p, _ := Mul(CumSum(x), w)
		return Sum(p)
	})
}

func TestGradCheckCumProd(t *testing.T) {
	w := New([]float64{0.5, -1, 2, 0.25}, []int{4})
	gradCheck(t, "cumprod", []float64{1.5, -0.7, 2, 0.4}, []int{4}, func(x *Tensor) *Tensor {
		p, _ := Mul(CumProd(x), w)
		return Sum(p)
	})
	// A zero in the series: the division-free backward pass must still be
	// exact where out_j / in_i would be 0/0.
	gradCheck(t, "cumprod-zero", []float64{2, 0, 3, 1.5}, []int{4}, func(x *Tensor) *Tensor {
		p, _ := Mul(CumProd(x), w)
		return Sum(p)
	})
}

func TestGradCheckCumMaxMin(t *testing.T) {
	// Values are distinct so the running extreme changes hands several times
	// and no finite-difference step lands on a tie.
	w := New([]float64{0.5, -1, 2, 0.25, 1.5}, []int{5})
	gradCheck(t, "cummax", []float64{1, 3, 2, 5, 4}, []int{5}, func(x *Tensor) *Tensor {
		p, _ := Mul(CumMax(x), w)
		return Sum(p)
	})
	gradCheck(t, "cummin", []float64{5, 3, 4, 1, 2}, []int{5}, func(x *Tensor) *Tensor {
		p, _ := Mul(CumMin(x), w)
		return Sum(p)
	})
}

func TestGradCheckSoftmax(t *testing.T) {
	// Weight softmax outputs so the scalar has a non-trivial gradient.
	w := New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int{2, 3})
	gradCheck(t, "softmax", []float64{1, 2, 0.5, -1, 0.3, 2}, []int{2, 3}, func(x *Tensor) *Tensor {
		s, _ := Softmax(x, 1)
		p, _ := Mul(s, w)
		return Sum(p)
	})
}

func TestGradCheckLogSumExp(t *testing.T) {
	gradCheck(t, "logsumexp", []float64{1, 2, 0.5, -1, 0.3, 2}, []int{2, 3}, func(x *Tensor) *Tensor {
		r, _ := LogSumExp(x, 1)
		return Sum(r)
	})
}

func TestGradCheckReshape(t *testing.T) {
	gradCheck(t, "reshape", []float64{1, 2, 3, 4, 5, 6}, []int{2, 3}, func(x *Tensor) *Tensor {
		r, _ := Reshape(x, []int{3, 2})
		return Sum(Square(r))
	})
}

func TestGradCheckTranspose(t *testing.T) {
	gradCheck(t, "transpose", []float64{1, 2, 3, 4, 5, 6}, []int{2, 3}, func(x *Tensor) *Tensor {
		r, _ := TransposePerm(x, nil)
		return Sum(Square(r))
	})
}

func TestGradCheckConcat(t *testing.T) {
	other := New([]float64{7, 8}, []int{1, 2})
	gradCheck(t, "concat", []float64{1, 2, 3, 4}, []int{2, 2}, func(x *Tensor) *Tensor {
		c, _ := Concat([]*Tensor{x, other}, 0)
		return Sum(Square(c))
	})
}

func TestGradCheckMatMul(t *testing.T) {
	b := New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2})
	gradCheck(t, "matmul", []float64{1, 0, -1, 2, 1, 1}, []int{2, 3}, func(x *Tensor) *Tensor {
		m, _ := MatMul(x, b)
		return Sum(Square(m))
	})
}
