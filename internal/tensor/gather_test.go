package tensor

import "testing"

func TestGatherValueAndShape(t *testing.T) {
	// x is 3 rows of width 2.
	x := New([]float64{10, 11, 20, 21, 30, 31}, []int{3, 2})
	y, err := Gather(x, []int{2, 0, 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(y.Shape) != 2 || y.Shape[0] != 3 || y.Shape[1] != 2 {
		t.Fatalf("shape = %v, want [3 2]", y.Shape)
	}
	want := []float64{30, 31, 10, 11, 30, 31}
	for i := range want {
		if y.Data[i] != want[i] {
			t.Fatalf("y = %v, want %v", y.Data, want)
		}
	}
}

func TestGatherOutOfRange(t *testing.T) {
	x := New([]float64{1, 2, 3, 4}, []int{2, 2})
	if _, err := Gather(x, []int{0, 2}); err == nil {
		t.Fatal("expected an out-of-range error")
	}
}

func TestGradCheckGather(t *testing.T) {
	// Repeated index (2 appears twice) exercises gradient scatter-accumulation.
	gradCheck(t, "gather", []float64{1, 2, 3, 4, 5, 6}, []int{3, 2}, func(x *Tensor) *Tensor {
		g, err := Gather(x, []int{2, 0, 2})
		if err != nil {
			t.Fatal(err)
		}
		return Sum(Square(g))
	})
}
