package interp_test

import "testing"

func TestVectorSlice(t *testing.T) {
	if got := scalar(t, "[10.0, 20.0, 30.0, 40.0][1:3][1]"); got != 30 {
		t.Errorf("got %v, want 30", got)
	}
	if got := scalar(t, "sum([10.0, 20.0, 30.0, 40.0][:2])"); got != 30 {
		t.Errorf("open-start slice got %v, want 30", got)
	}
	if got := scalar(t, "sum([10.0, 20.0, 30.0, 40.0][2:])"); got != 70 {
		t.Errorf("open-end slice got %v, want 70", got)
	}
}

func TestMatrixRowSlice(t *testing.T) {
	// m[1:3] keeps two rows; [0][1] of the result is m[1][1] = 4.
	if got := scalar(t, "[[1.0,2.0],[3.0,4.0],[5.0,6.0]][1:3][0][1]"); got != 4 {
		t.Errorf("got %v, want 4", got)
	}
}

func TestListSlice(t *testing.T) {
	// range gives a list; slicing it yields a list.
	if got := scalar(t, "range(10)[2:5][1]"); got != 3 {
		t.Errorf("got %v, want 3", got)
	}
}

func TestSliceIsDifferentiable(t *testing.T) {
	// d/dx sum(x[1:3]) routes gradient only to the sliced elements.
	src := "grad(fn(x) = sum(x[1:3]))([1.0, 2.0, 3.0, 4.0])[2]"
	if got := scalar(t, src); got != 1 {
		t.Errorf("got %v, want 1", got)
	}
	src2 := "grad(fn(x) = sum(x[1:3]))([1.0, 2.0, 3.0, 4.0])[0]"
	if got := scalar(t, src2); got != 0 {
		t.Errorf("got %v, want 0", got)
	}
}
