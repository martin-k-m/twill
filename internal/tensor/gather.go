package tensor

import "fmt"

// Gather selects rows from x along its first axis: with x shaped [N, ...D] and
// idx a list of row indices, the result is [len(idx), ...D]. It is
// differentiable — the backward pass scatter-adds each output row's gradient
// back to the source row, so repeated indices accumulate correctly. This is the
// primitive behind embedding lookups and index-based minibatching.
func Gather(x *Tensor, idx []int) (*Tensor, error) {
	if len(x.Shape) == 0 {
		return nil, fmt.Errorf("gather: cannot gather from a scalar")
	}
	n := x.Shape[0]
	rowSize := 1
	for _, d := range x.Shape[1:] {
		rowSize *= d
	}
	out := make([]float64, len(idx)*rowSize)
	for i, ix := range idx {
		if ix < 0 || ix >= n {
			return nil, fmt.Errorf("gather: index %d out of range [0, %d)", ix, n)
		}
		copy(out[i*rowSize:(i+1)*rowSize], x.Data[ix*rowSize:(ix+1)*rowSize])
	}
	outShape := make([]int, len(x.Shape))
	outShape[0] = len(idx)
	copy(outShape[1:], x.Shape[1:])
	res := &Tensor{Data: out, Shape: outShape}
	return track1(res, x, func() {
		gx := x.ensureGrad()
		g := res.Grad
		for i, ix := range idx {
			base, obase := ix*rowSize, i*rowSize
			for j := 0; j < rowSize; j++ {
				gx[base+j] += g[obase+j]
			}
		}
	}), nil
}
