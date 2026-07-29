package interp_test

import "testing"

// These reuse the run/scalar helpers from interp_test.go (same test package).

func TestBroadcastingRowVector(t *testing.T) {
	// A row vector broadcasts across the rows of a matrix.
	src := "let m = [[1.0, 2.0], [3.0, 4.0]] + [10.0, 20.0]\nm[1][1]"
	if got := scalar(t, src); got != 24 {
		t.Errorf("got %v, want 24", got)
	}
}

func TestBroadcastingColumnVector(t *testing.T) {
	src := "let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] * [[10.0], [100.0]]\nm[1][2]"
	if got := scalar(t, src); got != 600 {
		t.Errorf("got %v, want 600", got)
	}
}

func TestAxisReductions(t *testing.T) {
	if got := scalar(t, "sum([[1.0, 2.0], [3.0, 4.0]], 0)[1]"); got != 6 {
		t.Errorf("sum axis0 got %v", got)
	}
	if got := scalar(t, "mean([[1.0, 2.0], [3.0, 4.0]], 1)[0]"); got != 1.5 {
		t.Errorf("mean axis1 got %v", got)
	}
	if got := scalar(t, "max([[1.0, 9.0], [3.0, 4.0]], 0)[1]"); got != 9 {
		t.Errorf("max axis0 got %v", got)
	}
}

func TestSoftmaxSumsToOne(t *testing.T) {
	if got := scalar(t, "sum(softmax([1.0, 2.0, 3.0, 4.0], 0))"); got < 0.9999 || got > 1.0001 {
		t.Errorf("softmax sum got %v, want 1", got)
	}
}

func TestArgmax(t *testing.T) {
	if got := scalar(t, "argmax([3.0, 1.0, 9.0, 2.0], 0)"); got != 2 {
		t.Errorf("argmax got %v", got)
	}
}

func TestWhere(t *testing.T) {
	if got := scalar(t, "where([1.0, 0.0, 1.0], [7.0, 7.0, 7.0], [9.0, 9.0, 9.0])[1]"); got != 9 {
		t.Errorf("where got %v", got)
	}
}

func TestReshapeAndTranspose(t *testing.T) {
	if got := scalar(t, "reshape([1.0, 2.0, 3.0, 4.0], 2, 2)[1][0]"); got != 3 {
		t.Errorf("reshape got %v", got)
	}
	if got := scalar(t, "transpose([[1.0, 2.0], [3.0, 4.0]])[0][1]"); got != 3 {
		t.Errorf("transpose got %v", got)
	}
}

func TestConcat(t *testing.T) {
	src := "let a = [[1.0, 2.0]]\nlet b = [[3.0, 4.0]]\nconcat([a, b], 0)[1][0]"
	if got := scalar(t, src); got != 3 {
		t.Errorf("concat got %v", got)
	}
}

func TestFoldAndAppend(t *testing.T) {
	if got := scalar(t, "fold(fn(a, b) = a + b, 0.0, [1.0, 2.0, 3.0, 4.0])"); got != 10 {
		t.Errorf("fold got %v", got)
	}
	if got := scalar(t, "len(append([1.0, 2.0], 3.0))"); got != 3 {
		t.Errorf("append got %v", got)
	}
}

func TestGradThroughBroadcastAndSoftmax(t *testing.T) {
	// d/dx sum(softmax(x)) == 0 (softmax outputs always sum to 1).
	src := "grad(fn(x) = sum(softmax(x, 0)))([1.0, 2.0, 3.0])[0]"
	if got := scalar(t, src); got < -1e-9 || got > 1e-9 {
		t.Errorf("grad got %v, want ~0", got)
	}
}
