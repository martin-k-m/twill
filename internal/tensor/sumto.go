package tensor

import "fmt"

// SumTo reduces t back down to shape, which must broadcast up to t's shape. It
// is the inverse of a broadcast, and it is the operation the backward pass of
// every broadcasting elementwise op already performs inline: broadcastBinary's
// closure walks the output in flat index order and accumulates into the
// operand's index, which is exactly the loop below.
//
// It is exported because the codegen IR needs it as a named operation rather
// than as a thing buried in a closure. Naming it also fixes its summation
// order in one place: strictly increasing output index, accumulating into the
// target, matching what the interpreter's closures do element for element.
func SumTo(t *Tensor, shape []int) (*Tensor, error) {
	up, ok := BroadcastShape(shape, t.Shape)
	if !ok || !shapeEqual(up, t.Shape) {
		return nil, fmt.Errorf("sum_to: %v does not broadcast to %v", shape, t.Shape)
	}
	if shapeEqual(t.Shape, shape) {
		return t, nil
	}
	n := len(t.Data)
	out := make([]float64, numel(shape))
	// idx maps a flat output-space index of t.Shape to the flat index of the
	// smaller shape, using the same zero-stride trick effStrides gives the
	// forward pass.
	eff := effStrides(shape, t.Shape)
	ostr := strides(t.Shape)
	pos := make([]int, n)
	for o := 0; o < n; o++ {
		ii, rem := 0, o
		for d := 0; d < len(t.Shape); d++ {
			ii += (rem / ostr[d]) * eff[d]
			rem = rem % ostr[d]
		}
		pos[o] = ii
		out[ii] += t.Data[o]
	}
	res := (&Tensor{Data: out, Shape: append([]int(nil), shape...)}).withDTypeLike(t)
	return track1(res, t, func() {
		gt := t.ensureGrad()
		g := res.Grad
		for o := 0; o < n; o++ {
			gt[o] += g[pos[o]]
		}
	}), nil
}
