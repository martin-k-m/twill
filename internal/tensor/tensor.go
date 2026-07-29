// Package tensor is the differentiable tensor engine at the core of Raster.
//
// Every numeric value in Raster is a Tensor. Scalars are rank-0 tensors
// (empty shape). Operations build a reverse-mode autodiff graph, but only
// when an input requires gradients, so ordinary evaluation stays cheap.
package tensor

import (
	"fmt"
	"math"
)

type Tensor struct {
	Data         []float64
	Shape        []int
	RequiresGrad bool
	Grad         []float64
	// Autodiff parents. Most ops have one or two inputs, stored inline to avoid
	// a per-op slice allocation; pRest holds any beyond two (einsum, concat).
	p0, p1   *Tensor
	pRest    []*Tensor
	backward func()
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// New builds a tensor from data and a shape.
func New(data []float64, shape []int) *Tensor {
	return &Tensor{Data: data, Shape: shape}
}

// Scalar builds a rank-0 tensor.
func Scalar(x float64) *Tensor {
	return &Tensor{Data: []float64{x}, Shape: []int{}}
}

// Leaf builds a gradient-tracking copy of the given data/shape.
func Leaf(data []float64, shape []int) *Tensor {
	d := make([]float64, len(data))
	copy(d, data)
	s := make([]int, len(shape))
	copy(s, shape)
	return &Tensor{Data: d, Shape: s, RequiresGrad: true}
}

// Filled builds a tensor of the given shape with every element set to value.
func Filled(shape []int, value float64) *Tensor {
	d := make([]float64, numel(shape))
	for i := range d {
		d[i] = value
	}
	s := make([]int, len(shape))
	copy(s, shape)
	return &Tensor{Data: d, Shape: s}
}

func (t *Tensor) Size() int      { return numel(t.Shape) }
func (t *Tensor) IsScalar() bool { return len(t.Shape) == 0 }

// FromNested builds a tensor from an arbitrarily nested slice of float64.
func FromNested(value any) (*Tensor, error) {
	var shape []int
	probe := value
	for {
		s, ok := probe.([]any)
		if !ok {
			break
		}
		shape = append(shape, len(s))
		if len(s) == 0 {
			break
		}
		probe = s[0]
	}
	var flat []float64
	var walk func(v any, depth int) error
	walk = func(v any, depth int) error {
		if s, ok := v.([]any); ok {
			if depth >= len(shape) || len(s) != shape[depth] {
				return fmt.Errorf("ragged tensor literal: inconsistent row lengths")
			}
			for _, item := range s {
				if err := walk(item, depth+1); err != nil {
					return err
				}
			}
			return nil
		}
		if f, ok := v.(float64); ok {
			flat = append(flat, f)
			return nil
		}
		return fmt.Errorf("tensor literals may only contain numbers")
	}
	if err := walk(value, 0); err != nil {
		return nil, err
	}
	return &Tensor{Data: flat, Shape: shape}, nil
}

// ToNested returns the tensor as a nested []any / float64 structure.
func (t *Tensor) ToNested() any {
	if t.IsScalar() {
		return t.Data[0]
	}
	var build func(offset, dim int) (any, int)
	build = func(offset, dim int) (any, int) {
		if dim == len(t.Shape)-1 {
			row := make([]any, t.Shape[dim])
			for i := 0; i < t.Shape[dim]; i++ {
				row[i] = t.Data[offset+i]
			}
			return row, offset + t.Shape[dim]
		}
		rows := make([]any, t.Shape[dim])
		off := offset
		for i := 0; i < t.Shape[dim]; i++ {
			r, next := build(off, dim+1)
			rows[i] = r
			off = next
		}
		return rows, off
	}
	v, _ := build(0, 0)
	return v
}

func (t *Tensor) ensureGrad() []float64 {
	if t.Grad == nil {
		t.Grad = make([]float64, len(t.Data))
	}
	return t.Grad
}

// Backward runs reverse-mode backpropagation from a scalar output.
func (t *Tensor) Backward() error {
	if !t.IsScalar() {
		return fmt.Errorf("backward may only be called on a scalar output")
	}
	var topo []*Tensor
	seen := map[*Tensor]bool{}
	var visit func(n *Tensor)
	visit = func(n *Tensor) {
		if seen[n] {
			return
		}
		seen[n] = true
		if n.p0 != nil {
			visit(n.p0)
		}
		if n.p1 != nil {
			visit(n.p1)
		}
		for _, p := range n.pRest {
			visit(p)
		}
		topo = append(topo, n)
	}
	visit(t)
	t.ensureGrad()[0] = 1
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backward != nil {
			topo[i].backward()
		}
	}
	return nil
}

// track1 wires a single-input op into the autodiff graph when needed.
func track1(out, a *Tensor, backward func()) *Tensor {
	if a.RequiresGrad {
		out.RequiresGrad = true
		out.p0 = a
		out.backward = backward
	}
	return out
}

// track2 wires a two-input op into the autodiff graph when needed.
func track2(out, a, b *Tensor, backward func()) *Tensor {
	if a.RequiresGrad || b.RequiresGrad {
		out.RequiresGrad = true
		out.p0 = a
		out.p1 = b
		out.backward = backward
	}
	return out
}

// trackN wires an op with any number of inputs into the graph. Used by the few
// ops (einsum, concat) that take more than two operands.
func trackN(out *Tensor, prev []*Tensor, backward func()) *Tensor {
	rg := false
	for _, p := range prev {
		if p.RequiresGrad {
			rg = true
			break
		}
	}
	if rg {
		out.RequiresGrad = true
		if len(prev) > 0 {
			out.p0 = prev[0]
		}
		if len(prev) > 1 {
			out.p1 = prev[1]
		}
		if len(prev) > 2 {
			out.pRest = prev[2:]
		}
		out.backward = backward
	}
	return out
}

// strides returns the row-major strides for a shape.
func strides(shape []int) []int {
	s := make([]int, len(shape))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = acc
		acc *= shape[i]
	}
	return s
}

// BroadcastShape computes the NumPy-style broadcast of two shapes, aligning
// them from the right. Dimensions must be equal or one of them must be 1.
func BroadcastShape(a, b []int) ([]int, bool) {
	ra, rb := len(a), len(b)
	r := ra
	if rb > r {
		r = rb
	}
	out := make([]int, r)
	for i := 0; i < r; i++ {
		da, db := 1, 1
		if i < ra {
			da = a[ra-1-i]
		}
		if i < rb {
			db = b[rb-1-i]
		}
		switch {
		case da == db:
			out[r-1-i] = da
		case da == 1:
			out[r-1-i] = db
		case db == 1:
			out[r-1-i] = da
		default:
			return nil, false
		}
	}
	return out, true
}

// effStrides gives, for each dimension of outShape, the stride to advance in
// inShape's flat data — zero where inShape broadcasts (a size-1 or absent dim).
func effStrides(inShape, outShape []int) []int {
	rIn, rOut := len(inShape), len(outShape)
	real := strides(inShape)
	eff := make([]int, rOut)
	for i := 0; i < rOut; i++ {
		j := rOut - 1 - i
		if i < rIn && inShape[rIn-1-i] != 1 {
			eff[j] = real[rIn-1-i]
		} else {
			eff[j] = 0
		}
	}
	return eff
}

func broadcastBinary(a, b *Tensor, f func(x, y float64) float64,
	da func(x, y, o float64) float64, db func(x, y, o float64) float64) (*Tensor, error) {
	shape, ok := BroadcastShape(a.Shape, b.Shape)
	if !ok {
		return nil, fmt.Errorf("shape mismatch: cannot broadcast %v with %v", a.Shape, b.Shape)
	}
	n := numel(shape)
	out := make([]float64, n)
	// When no input needs a gradient, skip building the backward closure
	// entirely (parameter updates and other forward-only math hit this).
	rg := a.RequiresGrad || b.RequiresGrad

	// Fast paths for the common cases, avoiding index arithmetic entirely:
	// equal shapes, and a scalar on either side.
	switch {
	case shapeEqual(a.Shape, b.Shape):
		ad, bd := a.Data, b.Data
		for i := 0; i < n; i++ {
			out[i] = f(ad[i], bd[i])
		}
		res := &Tensor{Data: out, Shape: shape}
		if !rg {
			return res, nil
		}
		return track2(res, a, b, func() {
			g := res.Grad
			if a.RequiresGrad {
				ga := a.ensureGrad()
				for i := 0; i < n; i++ {
					ga[i] += da(ad[i], bd[i], out[i]) * g[i]
				}
			}
			if b.RequiresGrad {
				gb := b.ensureGrad()
				for i := 0; i < n; i++ {
					gb[i] += db(ad[i], bd[i], out[i]) * g[i]
				}
			}
		}), nil

	case len(b.Data) == 1: // scalar b broadcast over a
		ad, bs := a.Data, b.Data[0]
		for i := 0; i < n; i++ {
			out[i] = f(ad[i], bs)
		}
		res := &Tensor{Data: out, Shape: shape}
		if !rg {
			return res, nil
		}
		return track2(res, a, b, func() {
			g := res.Grad
			if a.RequiresGrad {
				ga := a.ensureGrad()
				for i := 0; i < n; i++ {
					ga[i] += da(ad[i], bs, out[i]) * g[i]
				}
			}
			if b.RequiresGrad {
				gb := b.ensureGrad()
				for i := 0; i < n; i++ {
					gb[0] += db(ad[i], bs, out[i]) * g[i]
				}
			}
		}), nil

	case len(a.Data) == 1: // scalar a broadcast over b
		as, bd := a.Data[0], b.Data
		for i := 0; i < n; i++ {
			out[i] = f(as, bd[i])
		}
		res := &Tensor{Data: out, Shape: shape}
		if !rg {
			return res, nil
		}
		return track2(res, a, b, func() {
			g := res.Grad
			if a.RequiresGrad {
				ga := a.ensureGrad()
				for i := 0; i < n; i++ {
					ga[0] += da(as, bd[i], out[i]) * g[i]
				}
			}
			if b.RequiresGrad {
				gb := b.ensureGrad()
				for i := 0; i < n; i++ {
					gb[i] += db(as, bd[i], out[i]) * g[i]
				}
			}
		}), nil
	}

	// General broadcasting. Walk the output with an odometer-style coordinate
	// counter so input offsets update incrementally — no division per element.
	rank := len(shape)
	effA := effStrides(a.Shape, shape)
	effB := effStrides(b.Shape, shape)
	coord := make([]int, rank)
	ia, ib := 0, 0
	// Record input offsets for the backward pass only when a gradient is needed.
	var iaHist, ibHist []int
	if rg {
		iaHist = make([]int, n)
		ibHist = make([]int, n)
	}
	for o := 0; o < n; o++ {
		if rg {
			iaHist[o], ibHist[o] = ia, ib
		}
		out[o] = f(a.Data[ia], b.Data[ib])
		for d := rank - 1; d >= 0; d-- {
			coord[d]++
			ia += effA[d]
			ib += effB[d]
			if coord[d] < shape[d] {
				break
			}
			coord[d] = 0
			ia -= effA[d] * shape[d]
			ib -= effB[d] * shape[d]
		}
	}
	res := &Tensor{Data: out, Shape: shape}
	if !rg {
		return res, nil
	}
	return track2(res, a, b, func() {
		g := res.Grad
		if a.RequiresGrad {
			ga := a.ensureGrad()
			for o := 0; o < n; o++ {
				ga[iaHist[o]] += da(a.Data[iaHist[o]], b.Data[ibHist[o]], out[o]) * g[o]
			}
		}
		if b.RequiresGrad {
			gb := b.ensureGrad()
			for o := 0; o < n; o++ {
				gb[ibHist[o]] += db(a.Data[iaHist[o]], b.Data[ibHist[o]], out[o]) * g[o]
			}
		}
	}), nil
}

func unary(a *Tensor, f func(x float64) float64, df func(x, o float64) float64) *Tensor {
	n := len(a.Data)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = f(a.Data[i])
	}
	res := &Tensor{Data: out, Shape: append([]int(nil), a.Shape...)}
	if !a.RequiresGrad {
		return res
	}
	return track1(res, a, func() {
		ga := a.ensureGrad()
		for i := 0; i < n; i++ {
			ga[i] += df(a.Data[i], out[i]) * res.Grad[i]
		}
	})
}

func Add(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float64) float64 { return x + y },
		func(x, y, o float64) float64 { return 1 }, func(x, y, o float64) float64 { return 1 })
}
func Sub(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float64) float64 { return x - y },
		func(x, y, o float64) float64 { return 1 }, func(x, y, o float64) float64 { return -1 })
}
func Mul(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float64) float64 { return x * y },
		func(x, y, o float64) float64 { return y }, func(x, y, o float64) float64 { return x })
}
func Div(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float64) float64 { return x / y },
		func(x, y, o float64) float64 { return 1 / y }, func(x, y, o float64) float64 { return -x / (y * y) })
}
func Mod(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float64) float64 { return x - math.Floor(x/y)*y },
		func(x, y, o float64) float64 { return 1 }, func(x, y, o float64) float64 { return -math.Floor(x / y) })
}

func PowScalar(a *Tensor, p float64) *Tensor {
	return unary(a, func(x float64) float64 { return math.Pow(x, p) },
		func(x, o float64) float64 { return p * math.Pow(x, p-1) })
}
func Neg(a *Tensor) *Tensor {
	return unary(a, func(x float64) float64 { return -x }, func(x, o float64) float64 { return -1 })
}
func Relu(a *Tensor) *Tensor {
	return unary(a, func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0
	}, func(x, o float64) float64 {
		if x > 0 {
			return 1
		}
		return 0
	})
}
func Exp(a *Tensor) *Tensor {
	return unary(a, math.Exp, func(x, o float64) float64 { return o })
}
func Log(a *Tensor) *Tensor {
	return unary(a, math.Log, func(x, o float64) float64 { return 1 / x })
}
func Sin(a *Tensor) *Tensor {
	return unary(a, math.Sin, func(x, o float64) float64 { return math.Cos(x) })
}
func Cos(a *Tensor) *Tensor {
	return unary(a, math.Cos, func(x, o float64) float64 { return -math.Sin(x) })
}
func Sqrt(a *Tensor) *Tensor {
	return unary(a, math.Sqrt, func(x, o float64) float64 { return 0.5 / o })
}
func Tanh(a *Tensor) *Tensor {
	return unary(a, math.Tanh, func(x, o float64) float64 { return 1 - o*o })
}
func Sigmoid(a *Tensor) *Tensor {
	return unary(a, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) },
		func(x, o float64) float64 { return o * (1 - o) })
}

func reduceAll(a *Tensor, mean bool) *Tensor {
	n := len(a.Data)
	s := 0.0
	for _, x := range a.Data {
		s += x
	}
	scale := 1.0
	if mean {
		scale = 1.0 / float64(n)
	}
	res := Scalar(s * scale)
	return track1(res, a, func() {
		if !a.RequiresGrad {
			return
		}
		ga := a.ensureGrad()
		g := res.Grad[0] * scale
		for i := 0; i < n; i++ {
			ga[i] += g
		}
	})
}

func Sum(a *Tensor) *Tensor  { return reduceAll(a, false) }
func Mean(a *Tensor) *Tensor { return reduceAll(a, true) }

func mm(a []float64, m, k int, b []float64, n int) []float64 {
	c := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			aip := a[i*k+p]
			if aip == 0 {
				continue
			}
			bRow := p * n
			cRow := i * n
			for j := 0; j < n; j++ {
				c[cRow+j] += aip * b[bRow+j]
			}
		}
	}
	return c
}

func transpose2d(a []float64, rows, cols int) []float64 {
	t := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			t[j*rows+i] = a[i*cols+j]
		}
	}
	return t
}

// MatMul multiplies 1-D or 2-D operands (the `@` operator).
func MatMul(a, b *Tensor) (*Tensor, error) {
	a2 := a.Shape
	if len(a.Shape) == 1 {
		a2 = []int{1, a.Shape[0]}
	}
	b2 := b.Shape
	if len(b.Shape) == 1 {
		b2 = []int{b.Shape[0], 1}
	}
	if len(a2) != 2 || len(b2) != 2 {
		return nil, fmt.Errorf("@ (matmul) requires 1-D or 2-D operands")
	}
	m, k := a2[0], a2[1]
	k2, n := b2[0], b2[1]
	if k != k2 {
		return nil, fmt.Errorf("shape mismatch in @: %v @ %v (inner %d != %d)", a.Shape, b.Shape, k, k2)
	}
	outData := mm(a.Data, m, k, b.Data, n)
	var outShape []int
	switch {
	case len(a.Shape) == 1 && len(b.Shape) == 1:
		outShape = []int{}
	case len(a.Shape) == 1:
		outShape = []int{n}
	case len(b.Shape) == 1:
		outShape = []int{m}
	default:
		outShape = []int{m, n}
	}
	res := &Tensor{Data: outData, Shape: outShape}
	return track2(res, a, b, func() {
		g := res.Grad // flat length m*n
		if a.RequiresGrad {
			bt := transpose2d(b.Data, k, n)
			dA := mm(g, m, n, bt, k)
			ga := a.ensureGrad()
			for i := range dA {
				ga[i] += dA[i]
			}
		}
		if b.RequiresGrad {
			at := transpose2d(a.Data, m, k)
			dB := mm(at, k, m, g, n)
			gb := b.ensureGrad()
			for i := range dB {
				gb[i] += dB[i]
			}
		}
	}), nil
}

// Transpose returns the transpose of a 2-D tensor (non-differentiable).
func Transpose(t *Tensor) (*Tensor, error) {
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("transpose expects a 2-D tensor")
	}
	r, c := t.Shape[0], t.Shape[1]
	return &Tensor{Data: transpose2d(t.Data, r, c), Shape: []int{c, r}}, nil
}
