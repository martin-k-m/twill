package tensor

import (
	"math"
	"testing"
)

// int4 block quantisation: ~13x smaller than f64, and QLinear4 stays within a
// few percent of the full product. Block scaling is what keeps 4 bits usable.
func TestQuantizeI4MemoryAndAccuracy(t *testing.T) {
	n, k, m := 128, 512, 32
	w := Leaf(randData(n*k, 11), []int{n, k})
	x := Leaf(randData(m*k, 12), []int{m, k})

	q, err := QuantizeI4(w, 32)
	if err != nil {
		t.Fatal(err)
	}

	f64Bytes := n * k * 8
	ratio := float64(f64Bytes) / float64(q.Bytes())
	if ratio < 9 {
		t.Fatalf("memory ratio %.2fx, want >= 9x", ratio)
	}
	t.Logf("memory %d -> %d bytes (%.2fx smaller)", f64Bytes, q.Bytes(), ratio)

	ref, _ := MatMulNT(x, w)
	got, _ := QLinear4(x, q)
	var sumAbsErr, sumAbsRef float64
	for i := range ref.Data {
		sumAbsErr += math.Abs(ref.Data[i] - got.Data[i])
		sumAbsRef += math.Abs(ref.Data[i])
	}
	relErr := sumAbsErr / sumAbsRef
	// 4-bit is inherently lossy; on random-uniform weights (the worst case, no
	// structure for the block scale to exploit) a few percent is expected, and
	// real Gaussian weights do better. int8 for comparison lands near 0.4%.
	if relErr > 0.08 {
		t.Fatalf("relative error %.4f, want <= 0.08", relErr)
	}
	t.Logf("relative error %.4f%%", relErr*100)
}

// getNibble must invert setNibble across the full signed 4-bit range and both
// nibble positions.
func TestNibbleRoundTrip(t *testing.T) {
	buf := make([]byte, 8)
	for _, code := range []int{-8, -7, -1, 0, 1, 7} {
		for _, p := range []int{0, 1, 4, 5} {
			setNibble(buf, 0, p, code)
			if got := getNibble(buf, 0, p); got != code {
				t.Fatalf("code %d at p %d round-tripped to %d", code, p, got)
			}
		}
	}
}
