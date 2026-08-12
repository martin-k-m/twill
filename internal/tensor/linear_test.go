package tensor

import "testing"

// MatMulNT(a,b) must be bit-identical to MatMul(a, Transpose2D(b)), primal and
// gradient, since dense_apply relies on that to keep every numeric test in place.
func TestMatMulNTMatchesTransposeMatMul(t *testing.T) {
	for _, dims := range [][3]int{{1, 5, 3}, {4, 7, 6}, {8, 8, 8}, {3, 1, 9}} {
		m, k, n := dims[0], dims[1], dims[2]
		a := Leaf(randData(m*k, 1), []int{m, k})
		w := Leaf(randData(n*k, 2), []int{n, k})
		a.RequiresGrad, w.RequiresGrad = true, true

		wt, _ := TransposePerm(w, []int{1, 0})
		ref, _ := MatMul(a, wt)
		got, _ := MatMulNT(a, w)

		if len(ref.Data) != len(got.Data) {
			t.Fatalf("shape %v vs %v", ref.Shape, got.Shape)
		}
		for i := range ref.Data {
			if ref.Data[i] != got.Data[i] {
				t.Fatalf("primal m=%d k=%d n=%d idx %d: %v != %v", m, k, n, i, ref.Data[i], got.Data[i])
			}
		}
	}
}

func randData(n, seed int) []float64 {
	d := make([]float64, n)
	x := uint64(seed*2654435761 + 1)
	for i := range d {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		d[i] = float64(int64(x%2000)-1000) / 500.0
	}
	return d
}

// Gradients through MatMulNT must match those through MatMul(a, Transposeᵀ).
func TestMatMulNTGradientMatches(t *testing.T) {
	m, k, n := 4, 5, 3
	mkGrads := func(useNT bool) ([]float64, []float64) {
		a := Leaf(randData(m*k, 3), []int{m, k})
		w := Leaf(randData(n*k, 4), []int{n, k})
		a.RequiresGrad, w.RequiresGrad = true, true
		var y *Tensor
		if useNT {
			y, _ = MatMulNT(a, w)
		} else {
			wt, _ := TransposePerm(w, []int{1, 0})
			y, _ = MatMul(a, wt)
		}
		s := Sum(y)
		if err := s.Backward(); err != nil {
			t.Fatal(err)
		}
		return a.Grad, w.Grad
	}
	aRef, wRef := mkGrads(false)
	aGot, wGot := mkGrads(true)
	for i := range aRef {
		if aRef[i] != aGot[i] {
			t.Fatalf("dA idx %d: %v != %v", i, aRef[i], aGot[i])
		}
	}
	for i := range wRef {
		if wRef[i] != wGot[i] {
			t.Fatalf("dW idx %d: %v != %v", i, wRef[i], wGot[i])
		}
	}
}
