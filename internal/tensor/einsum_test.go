package tensor

import "testing"

func TestEinsumMatmul(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := New([]float64{1, 0, 0, 1, 1, 1}, []int{3, 2})
	got, err := Einsum("ij,jk->ik", []*Tensor{a, b})
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := MatMul(a, b)
	if len(got.Data) != len(ref.Data) {
		t.Fatalf("shape %v vs %v", got.Shape, ref.Shape)
	}
	for i := range ref.Data {
		if got.Data[i] != ref.Data[i] {
			t.Fatalf("einsum matmul %v != %v", got.Data, ref.Data)
		}
	}
}

func TestEinsumTransposeAndSum(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	tr, _ := Einsum("ij->ji", []*Tensor{a})
	if tr.Shape[0] != 3 || tr.Shape[1] != 2 || tr.Data[1] != 4 {
		t.Fatalf("transpose got shape %v data %v", tr.Shape, tr.Data)
	}
	s, _ := Einsum("ij->", []*Tensor{a})
	if !s.IsScalar() || s.Data[0] != 21 {
		t.Fatalf("full sum got %v", s.Data)
	}
	rows, _ := Einsum("ij->i", []*Tensor{a})
	if rows.Shape[0] != 2 || rows.Data[0] != 6 || rows.Data[1] != 15 {
		t.Fatalf("row sums got %v", rows.Data)
	}
}

func TestEinsumImplicitOutput(t *testing.T) {
	// "ij,jk" with no arrow -> output labels appearing once: i,k -> "ik".
	a := New([]float64{1, 2, 3, 4}, []int{2, 2})
	b := New([]float64{1, 0, 0, 1}, []int{2, 2})
	got, _ := Einsum("ij,jk", []*Tensor{a, b})
	if got.Shape[0] != 2 || got.Shape[1] != 2 || got.Data[0] != 1 {
		t.Fatalf("implicit got shape %v data %v", got.Shape, got.Data)
	}
}

func TestGradCheckEinsumMatmul(t *testing.T) {
	b := New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2})
	gradCheck(t, "einsum-matmul", []float64{1, 0, -1, 2, 1, 1}, []int{2, 3}, func(x *Tensor) *Tensor {
		m, _ := Einsum("ij,jk->ik", []*Tensor{x, b})
		return Sum(Square(m))
	})
}

func TestGradCheckEinsumBilinear(t *testing.T) {
	// x_i W_ij y_j : gradient w.r.t. x.
	w := New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3})
	y := New([]float64{0.5, -1, 2}, []int{3})
	gradCheck(t, "einsum-bilinear", []float64{1, 2}, []int{2}, func(x *Tensor) *Tensor {
		return mustEinsum("i,ij,j->", x, w, y)
	})
}

func TestGradCheckEinsumWeightedContraction(t *testing.T) {
	// Gradient w.r.t. the weight in a contraction, output still a vector.
	x := New([]float64{1, -2, 0.5}, []int{3})
	gradCheck(t, "einsum-weight", []float64{1, 0, 2, 1, -1, 3}, []int{2, 3}, func(w *Tensor) *Tensor {
		v, _ := Einsum("ij,j->i", []*Tensor{w, x})
		return Sum(Square(v))
	})
}

func mustEinsum(spec string, ins ...*Tensor) *Tensor {
	r, err := Einsum(spec, ins)
	if err != nil {
		panic(err)
	}
	return r
}
