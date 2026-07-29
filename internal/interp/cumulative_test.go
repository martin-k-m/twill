package interp_test

import (
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
