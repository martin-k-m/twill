package interp_test

import (
	"math"
	"testing"

	"github.com/martin-k-m/raster/internal/tensor"
	"github.com/martin-k-m/raster/internal/value"
)

func TestCumulativeOps(t *testing.T) {
	cases := []struct {
		src  string
		want []float64
	}{
		{"cumsum([1.0, 2.0, 3.0, 4.0])", []float64{1, 3, 6, 10}},
		{"cumprod([1.0, 2.0, 3.0, 4.0])", []float64{1, 2, 6, 24}},
		{"cummax([1.0, 3.0, 2.0, 5.0, 4.0])", []float64{1, 3, 3, 5, 5}},
		{"cummin([5.0, 3.0, 4.0, 1.0, 2.0])", []float64{5, 3, 3, 1, 1}},
	}
	for _, c := range cases {
		v, _ := run(t, c.src)
		tv, ok := v.(*tensor.Tensor)
		if !ok {
			t.Fatalf("%s: expected a tensor, got %s", c.src, value.Format(v))
		}
		if len(tv.Data) != len(c.want) {
			t.Fatalf("%s: got length %d, want %d", c.src, len(tv.Data), len(c.want))
		}
		for i := range c.want {
			if tv.Data[i] != c.want[i] {
				t.Fatalf("%s: at index %d got %v, want %v", c.src, i, tv.Data[i], c.want[i])
			}
		}
	}
}

// Gradients through the scans, against the closed forms. The scans used to
// return an untracked tensor, so grad silently came back all zeros instead.
func TestCumulativeGradients(t *testing.T) {
	cases := []struct {
		src  string
		want []float64
	}{
		// d/dx_i of sum(cumsum(x)) is the number of outputs x_i reaches, n - i.
		{"grad(fn(x) = sum(cumsum(x)))([1.0, 2.0, 3.0, 4.0])", []float64{4, 3, 2, 1}},
		// Weighted, so a backward pass that accumulates the wrong way shows up:
		// d/dx_i of sum(w * cumsum(x)) is the sum of w from i onward.
		{"grad(fn(x) = sum([1.0, 10.0, 100.0] * cumsum(x)))([1.0, 2.0, 3.0])", []float64{111, 110, 100}},
		// sum(cumprod(x)) at [1,2,3,4]: prefix products are 1, 2, 6, 24, and
		// d/dx_i is the sum of the ones containing x_i, divided by x_i.
		{"grad(fn(x) = sum(cumprod(x)))([1.0, 2.0, 3.0, 4.0])", []float64{33, 16, 10, 6}},
		// A zero in the series: dividing the output by the input would give
		// 0/0 here, so only the division-free backward pass gets it right.
		{"grad(fn(x) = sum(cumprod(x)))([2.0, 0.0, 3.0, 1.0])", []float64{1, 14, 0, 0}},
		// cummax on an increasing series is the identity, so every input owns
		// its own output; cummin on the same series is x_0 repeated.
		{"grad(fn(x) = sum(cummax(x)))([1.0, 2.0, 3.0, 4.0])", []float64{1, 1, 1, 1}},
		{"grad(fn(x) = sum(cummin(x)))([1.0, 2.0, 3.0, 4.0])", []float64{4, 0, 0, 0}},
		// The running max changes hands at 3.0 and 5.0, so those two elements
		// take all of the gradient (2 outputs and 2 outputs respectively).
		{"grad(fn(x) = sum(cummax(x)))([1.0, 3.0, 2.0, 5.0, 4.0])", []float64{1, 2, 0, 2, 0}},
		{"grad(fn(x) = sum(cummin(x)))([5.0, 3.0, 4.0, 1.0, 2.0])", []float64{1, 2, 0, 2, 0}},
	}
	for _, c := range cases {
		v, _ := run(t, c.src)
		tv, ok := v.(*tensor.Tensor)
		if !ok {
			t.Fatalf("%s: expected a tensor, got %s", c.src, value.Format(v))
		}
		if len(tv.Data) != len(c.want) {
			t.Fatalf("%s: got length %d, want %d", c.src, len(tv.Data), len(c.want))
		}
		for i := range c.want {
			if math.Abs(tv.Data[i]-c.want[i]) > 1e-9 {
				t.Errorf("%s: at index %d got %v, want %v", c.src, i, tv.Data[i], c.want[i])
			}
		}
	}
}

// max_drawdown is built on cummax, so before the scans were differentiable its
// gradient was not merely zero but wrong: the peak term dropped out and only
// the trough term survived.
func TestCumulativeGradientThroughDrawdown(t *testing.T) {
	src := `import "../../std/backtest.ra" as bt
grad(fn(eq) = bt.max_drawdown(eq))([1.0, 1.2, 0.9, 1.5, 1.1])`
	v, _ := run(t, src)
	tv, ok := v.(*tensor.Tensor)
	if !ok {
		t.Fatalf("expected a tensor, got %s", value.Format(v))
	}
	// The worst drawdown is 1.1 against a peak of 1.5 (from index 3), so
	// d/d(peak) = trough/peak² and d/d(trough) = -1/peak.
	want := []float64{0, 0, 0, 1.1 / (1.5 * 1.5), -1 / 1.5}
	for i := range want {
		if math.Abs(tv.Data[i]-want[i]) > 1e-9 {
			t.Errorf("max_drawdown grad at index %d got %v, want %v", i, tv.Data[i], want[i])
		}
	}
}
