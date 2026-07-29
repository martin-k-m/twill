package tensor

import "testing"

// A fixed, non-degenerate kernel for input-gradient checks.
func fixedKernel() *Tensor {
	// [Cout=2, Cin=1, KH=2, KW=2]
	return New([]float64{
		0.5, -0.3, 0.2, 0.7,
		-0.1, 0.4, 0.6, -0.2,
	}, []int{2, 1, 2, 2})
}

func TestGradCheckConv2DInput(t *testing.T) {
	w := fixedKernel()
	// input [Cin=1, H=3, W=3]
	gradCheck(t, "conv2d-input", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{1, 3, 3}, func(x *Tensor) *Tensor {
		y, err := Conv2D(x, w)
		if err != nil {
			t.Fatal(err)
		}
		return Sum(y)
	})
}

func TestGradCheckConv2DWeight(t *testing.T) {
	// input fixed [Cin=1, H=3, W=3]; differentiate wrt the weight.
	in := New([]float64{0.2, -1, 0.5, 3, -2, 1, 0.7, 0.1, -0.4}, []int{1, 3, 3})
	gradCheck(t, "conv2d-weight", []float64{0.5, -0.3, 0.2, 0.7, -0.1, 0.4, 0.6, -0.2}, []int{2, 1, 2, 2}, func(k *Tensor) *Tensor {
		y, err := Conv2D(in, k)
		if err != nil {
			t.Fatal(err)
		}
		return Sum(y)
	})
}

func TestGradCheckConv2DMultiChannel(t *testing.T) {
	// input [Cin=2, H=3, W=3]; weight [Cout=2, Cin=2, KH=2, KW=2].
	w := New([]float64{
		0.5, -0.3, 0.2, 0.7, -0.1, 0.4, 0.6, -0.2,
		0.15, 0.25, -0.35, 0.45, -0.55, 0.65, 0.05, -0.85,
	}, []int{2, 2, 2, 2})
	data := []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9,
		-1, 0.5, 2, -3, 1.5, 0.2, 4, -2, 0.8,
	}
	gradCheck(t, "conv2d-multichannel", data, []int{2, 3, 3}, func(x *Tensor) *Tensor {
		y, err := Conv2D(x, w)
		if err != nil {
			t.Fatal(err)
		}
		return Sum(Square(y)) // a non-linear head, so gradients aren't all 1
	})
}

func TestGradCheckMaxPool2D(t *testing.T) {
	// Distinct values so no window has a tied maximum.
	data := []float64{
		1, 8, 2, 7,
		4, 3, 6, 5,
		9, 12, 10, 15,
		11, 14, 13, 16,
	}
	gradCheck(t, "maxpool2d", data, []int{1, 4, 4}, func(x *Tensor) *Tensor {
		y, err := MaxPool2D(x, 2)
		if err != nil {
			t.Fatal(err)
		}
		return Sum(Square(y))
	})
}

func TestConv2DShapeAndValue(t *testing.T) {
	// Single 1-channel 3x3 identity-ish check: a 2x2 averaging kernel.
	in := New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{1, 3, 3})
	w := New([]float64{1, 1, 1, 1}, []int{1, 1, 2, 2}) // sum of each 2x2 window
	y, err := Conv2D(in, w)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 2, 2}
	for i := range want {
		if y.Shape[i] != want[i] {
			t.Fatalf("shape = %v, want %v", y.Shape, want)
		}
	}
	// top-left window: 1+2+4+5 = 12
	if y.Data[0] != 12 {
		t.Fatalf("y[0] = %v, want 12", y.Data[0])
	}
	// bottom-right window: 5+6+8+9 = 28
	if y.Data[3] != 28 {
		t.Fatalf("y[3] = %v, want 28", y.Data[3])
	}
}
