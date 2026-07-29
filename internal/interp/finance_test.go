package interp_test

import (
	"math"
	"testing"

	"github.com/martin-k-m/raster/internal/tensor"
	"github.com/martin-k-m/raster/internal/value"
)

func listScalars(t *testing.T, v value.Value) []float64 {
	t.Helper()
	lst, ok := v.(*value.List)
	if !ok {
		t.Fatalf("expected a list, got %s", value.Format(v))
	}
	out := make([]float64, len(lst.Items))
	for i, it := range lst.Items {
		tv, ok := it.(*tensor.Tensor)
		if !ok {
			t.Fatalf("list item %d is not a tensor", i)
		}
		out[i] = tv.Data[0]
	}
	return out
}

func TestDeterministicByDefault(t *testing.T) {
	// A fresh interpreter seeds the same way each time, so results reproduce.
	a := scalar(t, "randn(5)[3]")
	b := scalar(t, "randn(5)[3]")
	if a != b {
		t.Errorf("randn not reproducible: %v vs %v", a, b)
	}
}

func TestSeedControlsRandomness(t *testing.T) {
	same1 := scalar(t, "seed(7)\nrandn(3)[0]")
	same2 := scalar(t, "seed(7)\nrandn(3)[0]")
	diff := scalar(t, "seed(8)\nrandn(3)[0]")
	if same1 != same2 {
		t.Errorf("same seed gave different values: %v vs %v", same1, same2)
	}
	if same1 == diff {
		t.Errorf("different seeds gave the same value: %v", same1)
	}
}

func TestMonteCarloOptionPricingAndGreeks(t *testing.T) {
	// Monte-Carlo European call; price and Greeks should match Black-Scholes
	// (S0=K=100, r=5%, vol=20%, T=1: price 10.4506, delta 0.6368, vega 37.524).
	prog := `
		seed(42)
		let Z = randn(200000)
		fn call(S0, K, r, sig, T) {
			let d = (r - 0.5 * sig * sig) * T
			let ST = S0 * exp(d + sig * sqrt(T) * Z)
			exp(-r * T) * mean(relu(ST - K))
		}
		let price = call(100.0, 100.0, 0.05, 0.2, 1.0)
		let delta = grad(fn(s) = call(s, 100.0, 0.05, 0.2, 1.0))(100.0)
		let vega = grad(fn(v) = call(100.0, 100.0, 0.05, v, 1.0))(0.2)
		[price, delta, vega]`
	v, _ := run(t, prog)
	got := listScalars(t, v)
	checks := []struct {
		name      string
		got, want float64
		tol       float64
	}{
		{"price", got[0], 10.4506, 0.1},
		{"delta", got[1], 0.6368, 0.01},
		{"vega", got[2], 37.524, 0.5},
	}
	for _, c := range checks {
		if math.Abs(c.got-c.want) > c.tol {
			t.Errorf("%s = %v, want ~%v (tol %v)", c.name, c.got, c.want, c.tol)
		}
	}
}
