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

func TestGradCheckProd(t *testing.T) {
	// Values kept away from zero: the derivative is continuous everywhere, but
	// a factor near zero makes the product's scale collapse and the central
	// difference lose all its significant digits.
	gradCheck(t, "prod", []float64{1.5, -2, 0.5, 3}, []int{4}, func(x *Tensor) *Tensor {
		return Prod(x)
	})
}

func TestGradCheckProdAxis(t *testing.T) {
	gradCheck(t, "prod-axis", []float64{1.5, -2, 0.5, 3, 2, -1}, []int{3, 2}, func(x *Tensor) *Tensor {
		r, _ := ProdAxis(x, 1)
		return Sum(r)
	})
}

// The zero cases are the ones the numeric check cannot reach, because a
// difference quotient at a zero factor still sees a product of zero on both
// sides for every *other* index. They are asserted directly instead.
func TestProdGradWithZeros(t *testing.T) {
	one := Leaf([]float64{0, 2, 3}, []int{3})
	if err := Prod(one).Backward(); err != nil {
		t.Fatal(err)
	}
	// Only the zero moves the product; the others are each multiplied by it.
	if got := one.Grad; got[0] != 6 || got[1] != 0 || got[2] != 0 {
		t.Errorf("one zero: grad = %v, want [6 0 0]", got)
	}

	two := Leaf([]float64{0, 0, 3}, []int{3})
	if err := Prod(two).Backward(); err != nil {
		t.Fatal(err)
	}
	for i, g := range two.Grad {
		if g != 0 {
			t.Errorf("two zeros: grad[%d] = %v, want 0", i, g)
		}
	}
}

func TestMedianValues(t *testing.T) {
	// Odd length picks the middle element; even averages the middle two.
	if got := Median(New([]float64{5, 1, 3}, []int{3})).Data[0]; got != 3 {
		t.Errorf("odd median = %v, want 3", got)
	}
	if got := Median(New([]float64{4, 1, 3, 2}, []int{4})).Data[0]; got != 2.5 {
		t.Errorf("even median = %v, want 2.5", got)
	}

	rows, err := MedianAxis(New([]float64{1, 2, 3, 4}, []int{2, 2}), 1)
	if err != nil {
		t.Fatal(err)
	}
	if rows.Data[0] != 1.5 || rows.Data[1] != 3.5 {
		t.Errorf("median axis 1 = %v, want [1.5 3.5]", rows.Data)
	}
	if len(rows.Shape) != 1 || rows.Shape[0] != 2 {
		t.Errorf("median axis 1 shape = %v, want [2]", rows.Shape)
	}
}

func TestMedianGrad(t *testing.T) {
	// Odd: all of it goes to the element that was selected.
	odd := Leaf([]float64{5, 1, 3}, []int{3})
	if err := Median(odd).Backward(); err != nil {
		t.Fatal(err)
	}
	if odd.Grad[0] != 0 || odd.Grad[1] != 0 || odd.Grad[2] != 1 {
		t.Errorf("odd grad = %v, want [0 0 1]", odd.Grad)
	}

	// Even: the two middle values share it, because the output is their mean.
	even := Leaf([]float64{4, 1, 3, 2}, []int{4})
	if err := Median(even).Backward(); err != nil {
		t.Fatal(err)
	}
	if even.Grad[0] != 0 || even.Grad[1] != 0 || even.Grad[2] != 0.5 || even.Grad[3] != 0.5 {
		t.Errorf("even grad = %v, want [0 0 0.5 0.5]", even.Grad)
	}
}

func TestReduceAxisBounds(t *testing.T) {
	x := New([]float64{1, 2, 3, 4}, []int{2, 2})
	if _, err := ProdAxis(x, 5); err == nil {
		t.Error("ProdAxis: out-of-range axis should error")
	}
	if _, err := MedianAxis(x, -5); err == nil {
		t.Error("MedianAxis: out-of-range axis should error")
	}
	// Negative axes resolve from the end, the same as every other reduction.
	r, err := MedianAxis(x, -1)
	if err != nil {
		t.Fatalf("MedianAxis(-1): %v", err)
	}
	if r.Data[0] != 1.5 || r.Data[1] != 3.5 {
		t.Errorf("MedianAxis(-1) = %v, want [1.5 3.5]", r.Data)
	}
}

func TestBroadcastToValues(t *testing.T) {
	// A missing leading axis is prepended.
	r, err := BroadcastTo(New([]float64{1, 2}, []int{2}), []int{3, 2})
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 2, 1, 2, 1, 2}
	for i := range want {
		if r.Data[i] != want[i] {
			t.Fatalf("prepend: got %v, want %v", r.Data, want)
		}
	}

	// A length-1 axis is stretched in place, which is the keepdims case.
	c, err := BroadcastTo(New([]float64{1, 2}, []int{2, 1}), []int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	wantCol := []float64{1, 1, 1, 2, 2, 2}
	for i := range wantCol {
		if c.Data[i] != wantCol[i] {
			t.Fatalf("column: got %v, want %v", c.Data, wantCol)
		}
	}

	// An already-matching shape is a copy, not an error.
	if _, err := BroadcastTo(New([]float64{1, 2}, []int{2}), []int{2}); err != nil {
		t.Errorf("identity broadcast: %v", err)
	}
}

func TestBroadcastToRejects(t *testing.T) {
	// A 3 cannot become a 4: only a 1 stretches.
	if _, err := BroadcastTo(New([]float64{1, 2, 3}, []int{3}), []int{4}); err == nil {
		t.Error("3 -> 4 should error")
	}
	// Broadcasting is one-way. [2,3] and [2,1] are compatible as operands, but
	// expanding [2,3] *to* [2,1] would have to shrink, so it is refused.
	if _, err := BroadcastTo(New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}), []int{2, 1}); err == nil {
		t.Error("[2,3] -> [2,1] should error")
	}
	// The target cannot have fewer axes than the input.
	if _, err := BroadcastTo(New([]float64{1, 2}, []int{1, 2}), []int{2}); err == nil {
		t.Error("dropping an axis should error")
	}
}

func TestGradCheckBroadcastTo(t *testing.T) {
	// Each input element is read three times, so its gradient is the sum over
	// the three outputs that used it.
	w := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	gradCheck(t, "broadcast-to", []float64{1, 2}, []int{2, 1}, func(x *Tensor) *Tensor {
		b, err := BroadcastTo(x, []int{2, 3})
		if err != nil {
			t.Fatal(err)
		}
		p, err := Mul(b, w)
		if err != nil {
			t.Fatal(err)
		}
		return Sum(p)
	})
}

func TestSplitRoundTrip(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 4})
	parts, err := Split(x, []int{1, 3}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("got %d pieces, want 2", len(parts))
	}
	if parts[0].Shape[1] != 1 || parts[1].Shape[1] != 3 {
		t.Errorf("piece shapes = %v %v", parts[0].Shape, parts[1].Shape)
	}
	if parts[1].Data[0] != 2 || parts[1].Data[3] != 6 {
		t.Errorf("second piece = %v", parts[1].Data)
	}

	// Concatenating on the same axis must give back exactly what went in.
	back, err := Concat(parts, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range x.Data {
		if back.Data[i] != x.Data[i] {
			t.Fatalf("round trip: got %v, want %v", back.Data, x.Data)
		}
	}
}

func TestSplitIsACopy(t *testing.T) {
	// Writing into a piece must not reach back into the parent.
	x := New([]float64{1, 2, 3, 4}, []int{4})
	parts, err := SplitEqual(x, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	parts[0].Data[0] = 99
	if x.Data[0] != 1 {
		t.Errorf("parent aliased: x.Data[0] = %v, want 1", x.Data[0])
	}
}

func TestSplitRejects(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	if _, err := Split(x, []int{1, 1}, 1); err == nil {
		t.Error("sizes summing short should error")
	}
	if _, err := Split(x, []int{2, 2}, 1); err == nil {
		t.Error("sizes summing long should error")
	}
	if _, err := Split(x, []int{-1, 4}, 1); err == nil {
		t.Error("a negative size should error")
	}
	if _, err := Split(x, nil, 1); err == nil {
		t.Error("no pieces should error")
	}
	// 3 does not divide by 2, and a ragged result is worse than an error.
	if _, err := SplitEqual(x, 2, 1); err == nil {
		t.Error("uneven equal split should error")
	}
	if _, err := SplitEqual(x, 0, 1); err == nil {
		t.Error("zero pieces should error")
	}
	if _, err := SplitEqual(x, 2, 7); err == nil {
		t.Error("out-of-range axis should error")
	}
}

func TestGradCheckSplit(t *testing.T) {
	// Both halves are used, with different weights, so a gradient that went to
	// the wrong piece or was dropped would show up.
	gradCheck(t, "split", []float64{1, 2, 3, 4}, []int{4}, func(x *Tensor) *Tensor {
		parts, err := SplitEqual(x, 2, 0)
		if err != nil {
			t.Fatal(err)
		}
		a, err := Mul(parts[0], New([]float64{10}, []int{1}))
		if err != nil {
			t.Fatal(err)
		}
		b, err := Mul(parts[1], New([]float64{-3}, []int{1}))
		if err != nil {
			t.Fatal(err)
		}
		s, err := Add(Sum(a), Sum(b))
		if err != nil {
			t.Fatal(err)
		}
		return s
	})
}

// --- sorting ---------------------------------------------------------------

func TestSortValues(t *testing.T) {
	x := New([]float64{3, 1, 2}, []int{3})
	up, err := SortAxis(x, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if up.Data[0] != 1 || up.Data[1] != 2 || up.Data[2] != 3 {
		t.Errorf("ascending = %v", up.Data)
	}
	down, err := SortAxis(x, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if down.Data[0] != 3 || down.Data[2] != 1 {
		t.Errorf("descending = %v", down.Data)
	}
}

func TestSortIsPerRunAlongTheAxis(t *testing.T) {
	// A matrix sorted on the last axis sorts each row, not the whole thing.
	x := New([]float64{3, 1, 2, 9, 7, 8}, []int{2, 3})
	rows, err := SortAxis(x, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 2, 3, 7, 8, 9}
	for i := range want {
		if rows.Data[i] != want[i] {
			t.Fatalf("rows = %v, want %v", rows.Data, want)
		}
	}

	cols, err := SortAxis(x, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	// Columns are already ordered, so this is the identity, which is the point:
	// the axis has to be respected rather than the buffer flattened.
	for i := range x.Data {
		if cols.Data[i] != x.Data[i] {
			t.Fatalf("columns = %v, want unchanged", cols.Data)
		}
	}
}

func TestSortGradientFollowsThePermutation(t *testing.T) {
	// The whole reason sort is differentiable: whatever gradient arrives at the
	// element now in a position belongs to whichever element started there.
	x := Leaf([]float64{3, 1, 2}, []int{3})
	sorted, err := SortAxis(x, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	weights := New([]float64{1, 10, 100}, []int{3})
	scaled, err := Mul(sorted, weights)
	if err != nil {
		t.Fatal(err)
	}
	if err := Sum(scaled).Backward(); err != nil {
		t.Fatal(err)
	}
	// 3 sorts last so it takes 100; 1 sorts first so it takes 1; 2 takes 10.
	want := []float64{100, 1, 10}
	for i, w := range want {
		if x.Grad[i] != w {
			t.Fatalf("grad = %v, want %v", x.Grad, want)
		}
	}
}

func TestSortIsStableOnTies(t *testing.T) {
	// An unstable sort returns the same values in a different arrangement, and
	// since the gradient follows the arrangement, a different gradient.
	x := Leaf([]float64{5, 5, 5}, []int{3})
	sorted, err := SortAxis(x, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	weights := New([]float64{1, 2, 4}, []int{3})
	scaled, err := Mul(sorted, weights)
	if err != nil {
		t.Fatal(err)
	}
	if err := Sum(scaled).Backward(); err != nil {
		t.Fatal(err)
	}
	for i, w := range []float64{1, 2, 4} {
		if x.Grad[i] != w {
			t.Fatalf("tie grad = %v, want each element to keep its own weight", x.Grad)
		}
	}
}

func TestGradCheckSort(t *testing.T) {
	// Values kept distinct: the derivative is a step function across a tie, so
	// a numeric check straddling one is comparing two different permutations.
	weights := New([]float64{1, 3, 7, 2}, []int{4})
	gradCheck(t, "sort", []float64{3, 1, 4, 2}, []int{4}, func(x *Tensor) *Tensor {
		sorted, err := SortAxis(x, 0, false)
		if err != nil {
			t.Fatal(err)
		}
		scaled, err := Mul(sorted, weights)
		if err != nil {
			t.Fatal(err)
		}
		return Sum(scaled)
	})
}

func TestArgsortGivesPositions(t *testing.T) {
	x := New([]float64{3, 1, 2}, []int{3})
	got, err := ArgsortAxis(x, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 2, 0}
	for i := range want {
		if got.Data[i] != want[i] {
			t.Fatalf("argsort = %v, want %v", got.Data, want)
		}
	}
}

func TestTopKKeepsTheLargestInOrder(t *testing.T) {
	x := New([]float64{3, 1, 4, 2}, []int{4})
	got, err := TopKAxis(x, 2, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != 2 || got.Data[0] != 4 || got.Data[1] != 3 {
		t.Errorf("topk = %v, want [4 3]", got.Data)
	}
	if len(got.Shape) != 1 || got.Shape[0] != 2 {
		t.Errorf("shape = %v, want [2]", got.Shape)
	}
}

func TestTopKSmallest(t *testing.T) {
	x := New([]float64{3, 1, 4, 2}, []int{4})
	got, err := TopKAxis(x, 2, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if got.Data[0] != 1 || got.Data[1] != 2 {
		t.Errorf("smallest = %v, want [1 2]", got.Data)
	}
}

func TestTopKShrinksOnlyItsAxis(t *testing.T) {
	x := New([]float64{3, 1, 2, 9, 7, 8}, []int{2, 3})
	got, err := TopKAxis(x, 2, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Shape) != 2 || got.Shape[0] != 2 || got.Shape[1] != 2 {
		t.Fatalf("shape = %v, want [2 2]", got.Shape)
	}
	want := []float64{3, 2, 9, 8}
	for i := range want {
		if got.Data[i] != want[i] {
			t.Fatalf("topk rows = %v, want %v", got.Data, want)
		}
	}
}

func TestTopKGradientSkipsWhatItDropped(t *testing.T) {
	// A value outside the top k does not move the output, so its gradient is
	// zero. That is correct rather than a simplification.
	x := Leaf([]float64{3, 1, 4, 2}, []int{4})
	top, err := TopKAxis(x, 2, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := Sum(top).Backward(); err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 0, 1, 0}
	for i := range want {
		if x.Grad[i] != want[i] {
			t.Fatalf("grad = %v, want %v", x.Grad, want)
		}
	}
}

func TestTopKRejectsAnImpossibleK(t *testing.T) {
	x := New([]float64{1, 2}, []int{2})
	if _, err := TopKAxis(x, 0, 0, true); err == nil {
		t.Error("k of zero should error")
	}
	if _, err := TopKAxis(x, 5, 0, true); err == nil {
		t.Error("k larger than the axis should error")
	}
	if _, err := ArgTopKAxis(x, 5, 0, true); err == nil {
		t.Error("argtopk should reject it too")
	}
}

func TestArgTopKGivesPositions(t *testing.T) {
	x := New([]float64{3, 1, 4, 2}, []int{4})
	got, err := ArgTopKAxis(x, 2, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if got.Data[0] != 2 || got.Data[1] != 0 {
		t.Errorf("argtopk = %v, want [2 0]", got.Data)
	}
}

func TestSortRejectsABadAxis(t *testing.T) {
	x := New([]float64{1, 2}, []int{2})
	if _, err := SortAxis(x, 3, false); err == nil {
		t.Error("out-of-range axis should error")
	}
	if _, err := ArgsortAxis(x, -5, false); err == nil {
		t.Error("out-of-range axis should error")
	}
}

// --- running totals --------------------------------------------------------

func TestCumsumValues(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	got, err := CumsumAxis(x, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 3, 6, 4, 9, 15}
	for i := range want {
		if got.Data[i] != want[i] {
			t.Fatalf("cumsum along rows = %v, want %v", got.Data, want)
		}
	}
	// The axis stays, unlike every reduction: each element gets an answer.
	if len(got.Shape) != 2 || got.Shape[0] != 2 || got.Shape[1] != 3 {
		t.Errorf("shape = %v, want [2 3]", got.Shape)
	}

	down, err := CumsumAxis(x, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantDown := []float64{1, 2, 3, 5, 7, 9}
	for i := range wantDown {
		if down.Data[i] != wantDown[i] {
			t.Fatalf("cumsum down columns = %v, want %v", down.Data, wantDown)
		}
	}
}

func TestCumprodValues(t *testing.T) {
	x := New([]float64{1, 2, 3, 2, 2, 2}, []int{2, 3})
	got, err := CumprodAxis(x, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 2, 6, 2, 4, 8}
	for i := range want {
		if got.Data[i] != want[i] {
			t.Fatalf("cumprod = %v, want %v", got.Data, want)
		}
	}
}

func TestCumulativeAxisIsNormalized(t *testing.T) {
	// -1 is the last axis, the way it is everywhere else.
	x := New([]float64{1, 2, 3, 4}, []int{2, 2})
	last, err := CumsumAxis(x, -1)
	if err != nil {
		t.Fatal(err)
	}
	explicit, _ := CumsumAxis(x, 1)
	for i := range last.Data {
		if last.Data[i] != explicit.Data[i] {
			t.Fatalf("axis -1 = %v, axis 1 = %v", last.Data, explicit.Data)
		}
	}
	if _, err := CumsumAxis(x, 2); err == nil {
		t.Error("an out-of-range axis was accepted")
	}
}

func TestGradCheckCumsum(t *testing.T) {
	// Weighted so the gradients differ per position: summing the running sums
	// unweighted gives every element the same gradient, which would pass even
	// if the backward pass were reversed.
	w := New([]float64{1, 2, 4, 8, 16, 32}, []int{2, 3})
	gradCheck(t, "cumsum", []float64{1, -2, 3, 0.5, -1.5, 2}, []int{2, 3}, func(x *Tensor) *Tensor {
		c, _ := CumsumAxis(x, 1)
		p, _ := Mul(c, w)
		return Sum(p)
	})
}

func TestGradCheckCumsumDownColumns(t *testing.T) {
	w := New([]float64{1, 2, 4, 8, 16, 32}, []int{2, 3})
	gradCheck(t, "cumsum-axis0", []float64{1, -2, 3, 0.5, -1.5, 2}, []int{2, 3}, func(x *Tensor) *Tensor {
		c, _ := CumsumAxis(x, 0)
		p, _ := Mul(c, w)
		return Sum(p)
	})
}

func TestGradCheckCumprod(t *testing.T) {
	w := New([]float64{1, 2, 4, 8, 16, 32}, []int{2, 3})
	gradCheck(t, "cumprod", []float64{1.5, -2, 3, 0.5, -1.5, 2}, []int{2, 3}, func(x *Tensor) *Tensor {
		c, _ := CumprodAxis(x, 1)
		p, _ := Mul(c, w)
		return Sum(p)
	})
}

// A zero anywhere in the run is where the obvious implementation of the
// cumprod gradient goes wrong: out[n]/x[k] is the product of the others only
// when x[k] is not zero, and zero is not a rare value for a tensor. The
// difference quotient reaches this one, since a zero factor still moves the
// products that come after it.
func TestGradCheckCumprodWithAZero(t *testing.T) {
	w := New([]float64{1, 2, 4, 8}, []int{4})
	gradCheck(t, "cumprod-zero", []float64{2, 0, 3, 4}, []int{4}, func(x *Tensor) *Tensor {
		c, _ := CumprodAxis(x, 0)
		p, _ := Mul(c, w)
		return Sum(p)
	})
}

func TestGradCheckCumprodWithTwoZeros(t *testing.T) {
	w := New([]float64{1, 2, 4, 8}, []int{4})
	gradCheck(t, "cumprod-two-zeros", []float64{2, 0, 0, 4}, []int{4}, func(x *Tensor) *Tensor {
		c, _ := CumprodAxis(x, 0)
		p, _ := Mul(c, w)
		return Sum(p)
	})
}

// --- argmin and flip -------------------------------------------------------

func TestArgminAndArgmax(t *testing.T) {
	x := New([]float64{3, 1, 2, 5, 9, 4}, []int{2, 3})
	mins, err := ArgminAxis(x, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Rows are [3 1 2] and [5 9 4]: the smallest sits at 1 and at 2.
	if mins.Data[0] != 1 || mins.Data[1] != 2 {
		t.Errorf("argmin = %v, want [1 2]", mins.Data)
	}
	maxes, _ := ArgmaxAxis(x, 1)
	if maxes.Data[0] != 0 || maxes.Data[1] != 1 {
		t.Errorf("argmax = %v, want [0 1]", maxes.Data)
	}
	// The axis is dropped, like every other reduction.
	if len(mins.Shape) != 1 || mins.Shape[0] != 2 {
		t.Errorf("shape = %v, want [2]", mins.Shape)
	}
}

func TestArgExtremeTiesGoToTheFirst(t *testing.T) {
	// The same rule as sort's stability and the cumulative scans. A tie rule
	// that differs between two operations moves an answer when a value is
	// merely repeated.
	x := New([]float64{2, 2, 1, 1}, []int{4})
	maxes, _ := ArgmaxAxis(x, 0)
	mins, _ := ArgminAxis(x, 0)
	if maxes.Data[0] != 0 {
		t.Errorf("argmax tie = %v, want 0", maxes.Data[0])
	}
	if mins.Data[0] != 2 {
		t.Errorf("argmin tie = %v, want 2", mins.Data[0])
	}
}

func TestFlipValues(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	rows, err := FlipAxis(x, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{3, 2, 1, 6, 5, 4}
	for i := range want {
		if rows.Data[i] != want[i] {
			t.Fatalf("flip rows = %v, want %v", rows.Data, want)
		}
	}
	cols, _ := FlipAxis(x, 0)
	wantCols := []float64{4, 5, 6, 1, 2, 3}
	for i := range wantCols {
		if cols.Data[i] != wantCols[i] {
			t.Fatalf("flip columns = %v, want %v", cols.Data, wantCols)
		}
	}
}

func TestFlipIsItsOwnInverse(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	once, _ := FlipAxis(x, 1)
	twice, _ := FlipAxis(once, 1)
	for i := range x.Data {
		if twice.Data[i] != x.Data[i] {
			t.Fatalf("flip twice = %v, want %v", twice.Data, x.Data)
		}
	}
}

func TestGradCheckFlip(t *testing.T) {
	// Weighted, so each position has a different gradient: an unweighted sum
	// would pass even if the backward pass forgot to reverse.
	w := New([]float64{1, 2, 4, 8, 16, 32}, []int{2, 3})
	gradCheck(t, "flip", []float64{1, -2, 3, 0.5, -1.5, 2}, []int{2, 3}, func(x *Tensor) *Tensor {
		f, _ := FlipAxis(x, 1)
		p, _ := Mul(f, w)
		return Sum(p)
	})
}
