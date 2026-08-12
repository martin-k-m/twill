package tensor

import (
	"math"
	"testing"
)

// The two claims that make quantisation worth having: the weight is ~8x smaller,
// and QLinear stays close to the full-precision product.
func TestQuantizeI8MemoryAndAccuracy(t *testing.T) {
	n, k, m := 128, 512, 32
	w := Leaf(randData(n*k, 7), []int{n, k})
	x := Leaf(randData(m*k, 8), []int{m, k})

	q, err := QuantizeI8(w)
	if err != nil {
		t.Fatal(err)
	}

	// Memory: the f64 matrix is n*k*8 bytes; the packed form is n*k + n*8.
	f64Bytes := n * k * 8
	ratio := float64(f64Bytes) / float64(q.Bytes())
	if ratio < 7.5 {
		t.Fatalf("memory ratio %.2fx, want >= 7.5x", ratio)
	}
	t.Logf("memory %d -> %d bytes (%.2fx smaller)", f64Bytes, q.Bytes(), ratio)

	// Accuracy: QLinear(x, q) vs the exact MatMulNT(x, w). Symmetric int8 has a
	// step of maxAbs/127, so the relative error of a dot product lands well under
	// 1%. Compare on the mean absolute error relative to the output magnitude.
	ref, _ := MatMulNT(x, w)
	got, _ := QLinear(x, q)
	var sumAbsErr, sumAbsRef float64
	for i := range ref.Data {
		sumAbsErr += math.Abs(ref.Data[i] - got.Data[i])
		sumAbsRef += math.Abs(ref.Data[i])
	}
	relErr := sumAbsErr / sumAbsRef
	if relErr > 0.02 {
		t.Fatalf("relative error %.4f, want <= 0.02", relErr)
	}
	t.Logf("relative error %.4f%%", relErr*100)
}

// Dequantize must invert the pack: dequantised weight matches the scale*Q it was
// built from, and re-quantising is a fixed point.
func TestDequantizeRoundTrip(t *testing.T) {
	n, k := 16, 40
	w := Leaf(randData(n*k, 9), []int{n, k})
	q, _ := QuantizeI8(w)
	dq := q.Dequantize()
	// Every dequantised element is within half a step of the original.
	for j := 0; j < n; j++ {
		step := q.Scale[j]
		for p := 0; p < k; p++ {
			e := math.Abs(dq.Data[j*k+p] - w.Data[j*k+p])
			if e > step/2+1e-12 {
				t.Fatalf("row %d col %d: error %v exceeds half-step %v", j, p, e, step/2)
			}
		}
	}
	// Quantising the dequantised weight reproduces the same codes.
	q2, _ := QuantizeI8(dq)
	for i := range q.Q {
		if q.Q[i] != q2.Q[i] {
			t.Fatalf("re-quantise changed code at %d: %d != %d", i, q.Q[i], q2.Q[i])
		}
	}
}
