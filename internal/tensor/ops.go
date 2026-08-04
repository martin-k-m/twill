package tensor

import (
	"fmt"
	"math"
	"sort"
)

// --- extra elementwise ops -------------------------------------------------

func Maximum(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b,
		func(x, y float64) float64 { return math.Max(x, y) },
		func(x, y, o float64) float64 {
			if x >= y {
				return 1
			}
			return 0
		},
		func(x, y, o float64) float64 {
			if x >= y {
				return 0
			}
			return 1
		},
		zero2, zero2, zero2)
}

func Minimum(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b,
		func(x, y float64) float64 { return math.Min(x, y) },
		func(x, y, o float64) float64 {
			if x <= y {
				return 1
			}
			return 0
		},
		func(x, y, o float64) float64 {
			if x <= y {
				return 0
			}
			return 1
		},
		zero2, zero2, zero2)
}

// compareOp builds a non-differentiable elementwise comparison returning 1/0.
func compareOp(name string, cmp func(x, y float64) bool) func(a, b *Tensor) (*Tensor, error) {
	return func(a, b *Tensor) (*Tensor, error) {
		return broadcastBinary(a, b,
			func(x, y float64) float64 {
				if cmp(x, y) {
					return 1
				}
				return 0
			},
			zero2, zero2, zero2, zero2, zero2)
	}
}

var (
	Greater      = compareOp("gt", func(x, y float64) bool { return x > y })
	Less         = compareOp("lt", func(x, y float64) bool { return x < y })
	GreaterEqual = compareOp("ge", func(x, y float64) bool { return x >= y })
	LessEqual    = compareOp("le", func(x, y float64) bool { return x <= y })
	EqualOp      = compareOp("eq", func(x, y float64) bool { return x == y })
)

func Square(a *Tensor) *Tensor {
	return unary(a, func(x float64) float64 { return x * x }, func(x, o float64) float64 { return 2 * x },
		func(x, o float64) float64 { return 2 })
}

// Clip clamps values into [lo, hi]; the gradient passes through only the
// interior.
func Clip(a *Tensor, lo, hi float64) *Tensor {
	return unary(a,
		func(x float64) float64 { return math.Min(math.Max(x, lo), hi) },
		func(x, o float64) float64 {
			if x > lo && x < hi {
				return 1
			}
			return 0
		},
		zeroU)
}

// Where selects a where cond is non-zero, else b (with broadcasting). It is
// not differentiable through cond.
func Where(cond, a, b *Tensor) (*Tensor, error) {
	s1, ok := BroadcastShape(cond.Shape, a.Shape)
	if !ok {
		return nil, fmt.Errorf("where: cannot broadcast %v with %v", cond.Shape, a.Shape)
	}
	shape, ok := BroadcastShape(s1, b.Shape)
	if !ok {
		return nil, fmt.Errorf("where: cannot broadcast %v with %v", s1, b.Shape)
	}
	n := numel(shape)
	ostr := strides(shape)
	effC := effStrides(cond.Shape, shape)
	effA := effStrides(a.Shape, shape)
	effB := effStrides(b.Shape, shape)
	idx := func(o int, eff []int) int {
		ii, rem := 0, o
		for d := 0; d < len(shape); d++ {
			ii += (rem / ostr[d]) * eff[d]
			rem = rem % ostr[d]
		}
		return ii
	}
	out := make([]float64, n)
	for o := 0; o < n; o++ {
		if cond.Data[idx(o, effC)] != 0 {
			out[o] = a.Data[idx(o, effA)]
		} else {
			out[o] = b.Data[idx(o, effB)]
		}
	}
	res := &Tensor{Data: out, Shape: shape}
	return track2(res, a, b, func() {
		g := res.Grad
		for o := 0; o < n; o++ {
			if cond.Data[idx(o, effC)] != 0 {
				if a.RequiresGrad {
					a.ensureGrad()[idx(o, effA)] += g[o]
				}
			} else if b.RequiresGrad {
				b.ensureGrad()[idx(o, effB)] += g[o]
			}
		}
	}), nil
}

// --- axis helpers ----------------------------------------------------------

// axisSpans returns (before, length, after) for reducing/scanning an axis,
// where flat index = i*length*after + k*after + j.
func axisSpans(shape []int, axis int) (before, length, after int) {
	before, length, after = 1, shape[axis], 1
	for i := 0; i < axis; i++ {
		before *= shape[i]
	}
	for i := axis + 1; i < len(shape); i++ {
		after *= shape[i]
	}
	return
}

func removeAxis(shape []int, axis int) []int {
	out := make([]int, 0, len(shape)-1)
	out = append(out, shape[:axis]...)
	out = append(out, shape[axis+1:]...)
	return out
}

func normalizeAxis(axis, rank int) (int, error) {
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return 0, fmt.Errorf("axis %d out of range for rank %d", axis, rank)
	}
	return axis, nil
}

func reduceAxis(t *Tensor, axis int, mean bool) (*Tensor, error) {
	axis, err := normalizeAxis(axis, len(t.Shape))
	if err != nil {
		return nil, err
	}
	before, L, after := axisSpans(t.Shape, axis)
	outN := before * after
	out := make([]float64, outN)
	scale := 1.0
	if mean {
		scale = 1.0 / float64(L)
	}
	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			s := 0.0
			base := i*L*after + j
			for k := 0; k < L; k++ {
				s += t.Data[base+k*after]
			}
			out[i*after+j] = s * scale
		}
	}
	res := &Tensor{Data: out, Shape: removeAxis(t.Shape, axis)}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for i := 0; i < before; i++ {
			for j := 0; j < after; j++ {
				gv := res.Grad[i*after+j] * scale
				base := i*L*after + j
				for k := 0; k < L; k++ {
					gt[base+k*after] += gv
				}
			}
		}
	}), nil
}

func SumAxis(t *Tensor, axis int) (*Tensor, error)  { return reduceAxis(t, axis, false) }
func MeanAxis(t *Tensor, axis int) (*Tensor, error) { return reduceAxis(t, axis, true) }

// extremeAxis implements max/min along an axis, routing gradient to the
// selected element.
func extremeAxis(t *Tensor, axis int, wantMax bool) (*Tensor, error) {
	axis, err := normalizeAxis(axis, len(t.Shape))
	if err != nil {
		return nil, err
	}
	before, L, after := axisSpans(t.Shape, axis)
	outN := before * after
	out := make([]float64, outN)
	argFlat := make([]int, outN)
	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			base := i*L*after + j
			best := t.Data[base]
			bestIdx := base
			for k := 1; k < L; k++ {
				v := t.Data[base+k*after]
				if (wantMax && v > best) || (!wantMax && v < best) {
					best, bestIdx = v, base+k*after
				}
			}
			out[i*after+j] = best
			argFlat[i*after+j] = bestIdx
		}
	}
	res := &Tensor{Data: out, Shape: removeAxis(t.Shape, axis)}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for o := 0; o < outN; o++ {
			gt[argFlat[o]] += res.Grad[o]
		}
	}), nil
}

func MaxAxis(t *Tensor, axis int) (*Tensor, error) { return extremeAxis(t, axis, true) }
func MinAxis(t *Tensor, axis int) (*Tensor, error) { return extremeAxis(t, axis, false) }

// ArgmaxAxis returns the index of the maximum along an axis (not differentiable).
func ArgmaxAxis(t *Tensor, axis int) (*Tensor, error) {
	axis, err := normalizeAxis(axis, len(t.Shape))
	if err != nil {
		return nil, err
	}
	before, L, after := axisSpans(t.Shape, axis)
	out := make([]float64, before*after)
	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			base := i*L*after + j
			best, bestK := t.Data[base], 0
			for k := 1; k < L; k++ {
				if v := t.Data[base+k*after]; v > best {
					best, bestK = v, k
				}
			}
			out[i*after+j] = float64(bestK)
		}
	}
	return &Tensor{Data: out, Shape: removeAxis(t.Shape, axis)}, nil
}

// MaxAll / MinAll reduce to a scalar.
func extremeAll(t *Tensor, wantMax bool) *Tensor {
	best, bestIdx := t.Data[0], 0
	for i, v := range t.Data {
		if (wantMax && v > best) || (!wantMax && v < best) {
			best, bestIdx = v, i
		}
	}
	res := Scalar(best)
	return track1(res, t, func() {
		if t.RequiresGrad {
			t.ensureGrad()[bestIdx] += res.Grad[0]
		}
	})
}

func MaxAll(t *Tensor) *Tensor { return extremeAll(t, true) }
func MinAll(t *Tensor) *Tensor { return extremeAll(t, false) }

// --- softmax and logsumexp -------------------------------------------------

// Softmax computes a numerically stable softmax along an axis.
func Softmax(t *Tensor, axis int) (*Tensor, error) {
	axis, err := normalizeAxis(axis, len(t.Shape))
	if err != nil {
		return nil, err
	}
	before, L, after := axisSpans(t.Shape, axis)
	out := make([]float64, len(t.Data))
	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			base := i*L*after + j
			m := t.Data[base]
			for k := 1; k < L; k++ {
				if v := t.Data[base+k*after]; v > m {
					m = v
				}
			}
			sum := 0.0
			for k := 0; k < L; k++ {
				e := math.Exp(t.Data[base+k*after] - m)
				out[base+k*after] = e
				sum += e
			}
			for k := 0; k < L; k++ {
				out[base+k*after] /= sum
			}
		}
	}
	res := &Tensor{Data: out, Shape: append([]int(nil), t.Shape...)}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for i := 0; i < before; i++ {
			for j := 0; j < after; j++ {
				base := i*L*after + j
				dot := 0.0
				for k := 0; k < L; k++ {
					dot += res.Grad[base+k*after] * out[base+k*after]
				}
				for k := 0; k < L; k++ {
					s := out[base+k*after]
					gt[base+k*after] += s * (res.Grad[base+k*after] - dot)
				}
			}
		}
	}), nil
}

// LogSumExp reduces an axis with a numerically stable log-sum-exp.
func LogSumExp(t *Tensor, axis int) (*Tensor, error) {
	axis, err := normalizeAxis(axis, len(t.Shape))
	if err != nil {
		return nil, err
	}
	before, L, after := axisSpans(t.Shape, axis)
	outN := before * after
	out := make([]float64, outN)
	soft := make([]float64, len(t.Data)) // softmax weights for the backward pass
	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			base := i*L*after + j
			m := t.Data[base]
			for k := 1; k < L; k++ {
				if v := t.Data[base+k*after]; v > m {
					m = v
				}
			}
			sum := 0.0
			for k := 0; k < L; k++ {
				e := math.Exp(t.Data[base+k*after] - m)
				soft[base+k*after] = e
				sum += e
			}
			out[i*after+j] = m + math.Log(sum)
			for k := 0; k < L; k++ {
				soft[base+k*after] /= sum
			}
		}
	}
	res := &Tensor{Data: out, Shape: removeAxis(t.Shape, axis)}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for i := 0; i < before; i++ {
			for j := 0; j < after; j++ {
				g := res.Grad[i*after+j]
				base := i*L*after + j
				for k := 0; k < L; k++ {
					gt[base+k*after] += soft[base+k*after] * g
				}
			}
		}
	}), nil
}

// --- shape ops -------------------------------------------------------------

// Reshape returns a tensor with the same data and a new shape (same size).
func Reshape(t *Tensor, shape []int) (*Tensor, error) {
	if numel(shape) != len(t.Data) {
		return nil, fmt.Errorf("reshape: cannot fit %d elements into shape %v", len(t.Data), shape)
	}
	data := make([]float64, len(t.Data))
	copy(data, t.Data)
	res := &Tensor{Data: data, Shape: append([]int(nil), shape...)}
	if recordJets && t.RequiresGrad {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			copy(res.jet.d, t.jet.d)
			copy(res.jet.dd, t.jet.dd)
		}
	}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for i := range gt {
			gt[i] += res.Grad[i]
		}
	}), nil
}

// TransposePerm permutes axes. With no axes it reverses them (a plain
// transpose).
func TransposePerm(t *Tensor, axes []int) (*Tensor, error) {
	r := len(t.Shape)
	if len(axes) == 0 {
		axes = make([]int, r)
		for i := range axes {
			axes[i] = r - 1 - i
		}
	}
	if len(axes) != r {
		return nil, fmt.Errorf("transpose: got %d axes for a rank-%d tensor", len(axes), r)
	}
	seen := make([]bool, r)
	outShape := make([]int, r)
	for i, ax := range axes {
		if ax < 0 || ax >= r || seen[ax] {
			return nil, fmt.Errorf("transpose: invalid axis permutation %v", axes)
		}
		seen[ax] = true
		outShape[i] = t.Shape[ax]
	}
	inStr := strides(t.Shape)
	outStr := strides(outShape)
	n := len(t.Data)
	out := make([]float64, n)
	// inIndex maps an output flat index back to the input flat index.
	inIndex := func(o int) int {
		rem, in := o, 0
		for i := 0; i < r; i++ {
			coord := rem / outStr[i]
			rem = rem % outStr[i]
			in += coord * inStr[axes[i]]
		}
		return in
	}
	for o := 0; o < n; o++ {
		out[o] = t.Data[inIndex(o)]
	}
	res := &Tensor{Data: out, Shape: outShape}
	if recordJets && t.RequiresGrad {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			for o := 0; o < n; o++ {
				src := inIndex(o)
				res.jet.d[o] = t.jet.d[src]
				res.jet.dd[o] = t.jet.dd[src]
			}
		}
	}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for o := 0; o < n; o++ {
			gt[inIndex(o)] += res.Grad[o]
		}
	}), nil
}

// IndexAxis0 selects element/row `idx` along the first axis, dropping that axis:
// a 1-D tensor yields a scalar, and an N-D tensor yields a row of shape
// t.Shape[1:]. It is differentiable — gradient flows to the selected row — and
// carries forward-mode tangents, so both grad and hessian pass through `x[i]`.
func IndexAxis0(t *Tensor, idx int) (*Tensor, error) {
	if len(t.Shape) == 0 {
		return nil, fmt.Errorf("cannot index a scalar")
	}
	dim0 := t.Shape[0]
	if idx < 0 || idx >= dim0 {
		return nil, fmt.Errorf("index %d out of range [0, %d)", idx, dim0)
	}
	rowSize := 1
	for _, d := range t.Shape[1:] {
		rowSize *= d
	}
	off := idx * rowSize
	data := make([]float64, rowSize)
	copy(data, t.Data[off:off+rowSize])
	res := &Tensor{Data: data, Shape: append([]int(nil), t.Shape[1:]...)}
	if recordJets && t.RequiresGrad {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			copy(res.jet.d, t.jet.d[off:off+rowSize])
			copy(res.jet.dd, t.jet.dd[off:off+rowSize])
		}
	}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for i := 0; i < rowSize; i++ {
			gt[off+i] += res.Grad[i]
		}
	}), nil
}

// SliceAxis0 returns rows [start, end) along the first axis, keeping the
// trailing dimensions. It is differentiable: gradient flows to the selected
// rows.
func SliceAxis0(t *Tensor, start, end int) (*Tensor, error) {
	if len(t.Shape) == 0 {
		return nil, fmt.Errorf("cannot slice a scalar")
	}
	dim0 := t.Shape[0]
	if start < 0 {
		start += dim0
	}
	if end < 0 {
		end += dim0
	}
	if start < 0 || end > dim0 || start > end {
		return nil, fmt.Errorf("slice [%d:%d] out of range for first dim %d", start, end, dim0)
	}
	rowSize := 1
	for _, d := range t.Shape[1:] {
		rowSize *= d
	}
	outShape := append([]int{end - start}, t.Shape[1:]...)
	data := make([]float64, (end-start)*rowSize)
	copy(data, t.Data[start*rowSize:end*rowSize])
	res := &Tensor{Data: data, Shape: outShape}
	if recordJets && t.RequiresGrad {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			off := start * rowSize
			copy(res.jet.d, t.jet.d[off:off+len(data)])
			copy(res.jet.dd, t.jet.dd[off:off+len(data)])
		}
	}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for i := 0; i < len(data); i++ {
			gt[start*rowSize+i] += res.Grad[i]
		}
	}), nil
}

// Concat joins tensors along an axis. Shapes must match on every other axis.
func Concat(tensors []*Tensor, axis int) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("concat: need at least one tensor")
	}
	rank := len(tensors[0].Shape)
	axis, err := normalizeAxis(axis, rank)
	if err != nil {
		return nil, err
	}
	outShape := append([]int(nil), tensors[0].Shape...)
	total := 0
	for _, t := range tensors {
		if len(t.Shape) != rank {
			return nil, fmt.Errorf("concat: mismatched ranks")
		}
		for d := 0; d < rank; d++ {
			if d != axis && t.Shape[d] != tensors[0].Shape[d] {
				return nil, fmt.Errorf("concat: shapes differ on axis %d", d)
			}
		}
		total += t.Shape[axis]
	}
	outShape[axis] = total
	before, _, after := axisSpans(outShape, axis)
	out := make([]float64, numel(outShape))
	// Copy each input into its slice of the output axis.
	type placement struct {
		t      *Tensor
		offset int
	}
	var places []placement
	off := 0
	for _, t := range tensors {
		places = append(places, placement{t, off})
		off += t.Shape[axis]
	}
	for _, pl := range places {
		L := pl.t.Shape[axis]
		for i := 0; i < before; i++ {
			for k := 0; k < L; k++ {
				for j := 0; j < after; j++ {
					out[i*total*after+(pl.offset+k)*after+j] = pl.t.Data[i*L*after+k*after+j]
				}
			}
		}
	}
	prev := make([]*Tensor, len(tensors))
	copy(prev, tensors)
	res := &Tensor{Data: out, Shape: outShape}
	anyRG := false
	for _, t := range tensors {
		if t.RequiresGrad {
			anyRG = true
			break
		}
	}
	if recordJets && anyRG {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			for _, pl := range places {
				L := pl.t.Shape[axis]
				for i := 0; i < before; i++ {
					for k := 0; k < L; k++ {
						for j := 0; j < after; j++ {
							dst := i*total*after + (pl.offset+k)*after + j
							src := i*L*after + k*after + j
							res.jet.d[dst] = pl.t.jet.d[src]
							res.jet.dd[dst] = pl.t.jet.dd[src]
						}
					}
				}
			}
		}
	}
	return trackN(res, prev, func() {
		for _, pl := range places {
			if !pl.t.RequiresGrad {
				continue
			}
			gt := pl.t.ensureGrad()
			L := pl.t.Shape[axis]
			for i := 0; i < before; i++ {
				for k := 0; k < L; k++ {
					for j := 0; j < after; j++ {
						gt[i*L*after+k*after+j] += res.Grad[i*total*after+(pl.offset+k)*after+j]
					}
				}
			}
		}
	}), nil
}

// --- product and median ----------------------------------------------------

// Prod multiplies every element together.
func Prod(t *Tensor) *Tensor { return prodOver(t, len(t.Data), 1, 1, []int{}) }

// ProdAxis multiplies along one axis, dropping it from the shape.
func ProdAxis(t *Tensor, axis int) (*Tensor, error) {
	axis, err := normalizeAxis(axis, len(t.Shape))
	if err != nil {
		return nil, err
	}
	before, L, after := axisSpans(t.Shape, axis)
	return prodOver(t, L, before, after, removeAxis(t.Shape, axis)), nil
}

// prodOver is the shared body: `before * after` independent runs of length L,
// strided by `after`. Passing before=after=1 and L=len(Data) reduces the whole
// tensor, so the two entry points above are the same code.
//
// The derivative of a product with respect to one factor is the product of the
// others, which is the whole product divided by that factor — except that the
// division is invalid exactly when the factor is zero, and that is not a rare
// case for a tensor. So the zeros are counted instead:
//
//	no zeros    every factor gets total/x
//	one zero    that factor gets the product of the rest; everything else 0
//	two or more every derivative is 0, because every product of the others
//	            still contains a zero
func prodOver(t *Tensor, L, before, after int, outShape []int) *Tensor {
	outN := before * after
	out := make([]float64, outN)
	// Product of the non-zero factors, and how many zeros, per run. Both are
	// needed by the backward pass and are free to keep while multiplying.
	nzProd := make([]float64, outN)
	zeros := make([]int, outN)

	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			base := i*L*after + j
			p := 1.0
			nz := 1.0
			z := 0
			for k := 0; k < L; k++ {
				v := t.Data[base+k*after]
				p *= v
				if v == 0 {
					z++
				} else {
					nz *= v
				}
			}
			idx := i*after + j
			out[idx] = p
			nzProd[idx] = nz
			zeros[idx] = z
		}
	}

	res := &Tensor{Data: out, Shape: outShape}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for i := 0; i < before; i++ {
			for j := 0; j < after; j++ {
				idx := i*after + j
				g := res.Grad[idx]
				if g == 0 {
					continue
				}
				base := i*L*after + j
				switch zeros[idx] {
				case 0:
					for k := 0; k < L; k++ {
						p := base + k*after
						gt[p] += g * out[idx] / t.Data[p]
					}
				case 1:
					// Only the zero factor has a non-zero derivative: every
					// other one's "product of the rest" still contains it.
					for k := 0; k < L; k++ {
						p := base + k*after
						if t.Data[p] == 0 {
							gt[p] += g * nzProd[idx]
						}
					}
				}
			}
		}
	})
}

// Median returns the median of every element.
func Median(t *Tensor) *Tensor { return medianOver(t, len(t.Data), 1, 1, []int{}) }

// MedianAxis takes the median along one axis, dropping it from the shape.
func MedianAxis(t *Tensor, axis int) (*Tensor, error) {
	axis, err := normalizeAxis(axis, len(t.Shape))
	if err != nil {
		return nil, err
	}
	before, L, after := axisSpans(t.Shape, axis)
	return medianOver(t, L, before, after, removeAxis(t.Shape, axis)), nil
}

// medianOver mirrors prodOver's layout.
//
// The gradient routes to whichever element was selected, the same way max and
// min do: the median of a run is one of its values, so moving any other value
// a little does not move the answer at all. An even-length run averages the two
// middle values, so each of those takes half.
//
// Selection is by sorting indices rather than values, because the backward pass
// needs to know *which* element was chosen, not just what it held.
func medianOver(t *Tensor, L, before, after int, outShape []int) *Tensor {
	outN := before * after
	out := make([]float64, outN)
	// One or two source offsets per output, and the weight each carries back.
	lo := make([]int, outN)
	hi := make([]int, outN)

	idxBuf := make([]int, L)
	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			base := i*L*after + j
			for k := 0; k < L; k++ {
				idxBuf[k] = base + k*after
			}
			sort.Slice(idxBuf, func(a, b int) bool {
				return t.Data[idxBuf[a]] < t.Data[idxBuf[b]]
			})
			idx := i*after + j
			mid := L / 2
			if L%2 == 1 {
				lo[idx] = idxBuf[mid]
				hi[idx] = idxBuf[mid]
				out[idx] = t.Data[idxBuf[mid]]
			} else {
				lo[idx] = idxBuf[mid-1]
				hi[idx] = idxBuf[mid]
				out[idx] = (t.Data[idxBuf[mid-1]] + t.Data[idxBuf[mid]]) / 2
			}
		}
	}

	res := &Tensor{Data: out, Shape: outShape}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for idx := 0; idx < outN; idx++ {
			g := res.Grad[idx]
			if g == 0 {
				continue
			}
			if lo[idx] == hi[idx] {
				gt[lo[idx]] += g
			} else {
				gt[lo[idx]] += g / 2
				gt[hi[idx]] += g / 2
			}
		}
	})
}
