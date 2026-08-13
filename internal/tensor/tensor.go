// Package tensor is the differentiable tensor engine at the core of Twill.
//
// Every numeric value in Twill is a Tensor. Scalars are rank-0 tensors
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
	// Forward-mode state, allocated only during a second-derivative computation
	// (nil in normal training and grad), so the common Tensor stays small.
	jet *jetState
	// Element type, stored as the code plus one so the zero value -- what the
	// many &Tensor{...} intermediates get -- reads as f64, the default, without
	// colliding with bool (code 0). Read it through DType, never directly.
	dtp uint8
}

// DType reports the tensor's element type. The zero value of the internal field
// means f64, so an intermediate built without naming a dtype is f64, matching
// the default for every tensor that existed before dtypes.
func (t *Tensor) DType() DType {
	if t.dtp == 0 {
		return DTF64
	}
	return DType(t.dtp - 1)
}

// WithDType stamps a dtype onto the tensor in place and returns it, for the
// constructors and casts that produce a value of a named dtype. It records only
// the tag; the caller is responsible for the buffer invariant, that every
// element is already the f64 widening of a value of dt.
func (t *Tensor) WithDType(dt DType) *Tensor {
	t.dtp = uint8(dt) + 1
	return t
}

// jetState holds forward-mode (jet) data for one node: the first and second
// directional derivatives along a seed direction, and the closure that
// recomputes them from the node's parents. See jet.go.
type jetState struct {
	d, dd []float64
	jvp   func()
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
	da func(x, y, o float64) float64, db func(x, y, o float64) float64,
	daa, dab, dbb func(x, y, o float64) float64) (*Tensor, error) {
	// Two scalars with nothing to differentiate is what an interpreted loop is
	// made of, and the general path charges it a broadcast computation, a
	// parallelFor over a single element, and a slice for that element. Doing the
	// arithmetic is cheaper than deciding how to do it.
	//
	// Gated on gradients and jets being off rather than handled here, because
	// the backward closure is the part worth having exactly one copy of.
	if !recordJets && !a.RequiresGrad && !b.RequiresGrad &&
		len(a.Shape) == 0 && len(b.Shape) == 0 {
		return Scalar(f(a.Data[0], b.Data[0])), nil
	}

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
		parallelFor(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				out[i] = f(ad[i], bd[i])
			}
		})
		res := &Tensor{Data: out, Shape: shape}
		if !rg {
			return res, nil
		}
		if recordJets {
			res.jet = &jetState{jvp: jetBinary(res, a, b, da, db, daa, dab, dbb)}
		}
		return track2(res, a, b, func() {
			g := res.Grad
			if a.RequiresGrad {
				ga := a.ensureGrad()
				parallelFor(n, func(lo, hi int) {
					for i := lo; i < hi; i++ {
						ga[i] += da(ad[i], bd[i], out[i]) * g[i]
					}
				})
			}
			if b.RequiresGrad {
				gb := b.ensureGrad()
				parallelFor(n, func(lo, hi int) {
					for i := lo; i < hi; i++ {
						gb[i] += db(ad[i], bd[i], out[i]) * g[i]
					}
				})
			}
		}), nil

	case len(b.Data) == 1: // scalar b broadcast over a
		ad, bs := a.Data, b.Data[0]
		parallelFor(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				out[i] = f(ad[i], bs)
			}
		})
		res := &Tensor{Data: out, Shape: shape}
		if !rg {
			return res, nil
		}
		if recordJets {
			res.jet = &jetState{jvp: jetBinary(res, a, b, da, db, daa, dab, dbb)}
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
		parallelFor(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				out[i] = f(as, bd[i])
			}
		})
		res := &Tensor{Data: out, Shape: shape}
		if !rg {
			return res, nil
		}
		if recordJets {
			res.jet = &jetState{jvp: jetBinary(res, a, b, da, db, daa, dab, dbb)}
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
	if recordJets {
		res.jet = &jetState{jvp: jetBinary(res, a, b, da, db, daa, dab, dbb)}
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

func unary(a *Tensor, f func(x float64) float64, df func(x, o float64) float64, ddf func(x, o float64) float64) *Tensor {
	n := len(a.Data)
	out := make([]float64, n)
	ad := a.Data
	parallelFor(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = f(ad[i])
		}
	})
	res := &Tensor{Data: out, Shape: append([]int(nil), a.Shape...)}
	if !a.RequiresGrad {
		return res
	}
	if recordJets {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			rd, rdd, ad1, add := res.jet.d, res.jet.dd, a.jet.d, a.jet.dd
			for i := 0; i < n; i++ {
				d1 := df(ad[i], out[i])
				rd[i] = d1 * ad1[i]
				rdd[i] = d1*add[i] + ddf(ad[i], out[i])*ad1[i]*ad1[i]
			}
		}
	}
	return track1(res, a, func() {
		ga := a.ensureGrad()
		g := res.Grad
		parallelFor(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				ga[i] += df(ad[i], out[i]) * g[i]
			}
		})
	})
}

// zero2 is the constant-zero second derivative used by ops that are (locally)
// linear in an argument.
func zero2(x, y, o float64) float64 { return 0 }

func Add(a, b *Tensor) (*Tensor, error) {
	one := func(x, y, o float64) float64 { return 1 }
	return broadcastBinary(a, b, func(x, y float64) float64 { return x + y },
		one, one, zero2, zero2, zero2)
}
func Sub(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float64) float64 { return x - y },
		func(x, y, o float64) float64 { return 1 }, func(x, y, o float64) float64 { return -1 },
		zero2, zero2, zero2)
}
func Mul(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float64) float64 { return x * y },
		func(x, y, o float64) float64 { return y }, func(x, y, o float64) float64 { return x },
		zero2, func(x, y, o float64) float64 { return 1 }, zero2)
}
func Div(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float64) float64 { return x / y },
		func(x, y, o float64) float64 { return 1 / y }, func(x, y, o float64) float64 { return -x / (y * y) },
		zero2,
		func(x, y, o float64) float64 { return -1 / (y * y) },
		func(x, y, o float64) float64 { return 2 * x / (y * y * y) })
}
func Mod(a, b *Tensor) (*Tensor, error) {
	return broadcastBinary(a, b, func(x, y float64) float64 { return x - math.Floor(x/y)*y },
		func(x, y, o float64) float64 { return 1 }, func(x, y, o float64) float64 { return -math.Floor(x / y) },
		zero2, zero2, zero2)
}

// zeroU is the constant-zero second derivative for (locally) linear unary ops.
func zeroU(x, o float64) float64 { return 0 }

func PowScalar(a *Tensor, p float64) *Tensor {
	return unary(a, func(x float64) float64 { return math.Pow(x, p) },
		func(x, o float64) float64 { return p * math.Pow(x, p-1) },
		func(x, o float64) float64 { return p * (p - 1) * math.Pow(x, p-2) })
}
func Neg(a *Tensor) *Tensor {
	return unary(a, func(x float64) float64 { return -x }, func(x, o float64) float64 { return -1 }, zeroU)
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
	}, zeroU)
}
func Exp(a *Tensor) *Tensor {
	return unary(a, math.Exp, func(x, o float64) float64 { return o }, func(x, o float64) float64 { return o })
}
func Log(a *Tensor) *Tensor {
	return unary(a, math.Log, func(x, o float64) float64 { return 1 / x }, func(x, o float64) float64 { return -1 / (x * x) })
}
func Sin(a *Tensor) *Tensor {
	return unary(a, math.Sin, func(x, o float64) float64 { return math.Cos(x) }, func(x, o float64) float64 { return -o })
}
func Cos(a *Tensor) *Tensor {
	return unary(a, math.Cos, func(x, o float64) float64 { return -math.Sin(x) }, func(x, o float64) float64 { return -o })
}
func Sqrt(a *Tensor) *Tensor {
	return unary(a, math.Sqrt, func(x, o float64) float64 { return 0.5 / o }, func(x, o float64) float64 { return -0.25 / (o * o * o) })
}
func Tanh(a *Tensor) *Tensor {
	return unary(a, math.Tanh, func(x, o float64) float64 { return 1 - o*o }, func(x, o float64) float64 { return -2 * o * (1 - o*o) })
}
func Sigmoid(a *Tensor) *Tensor {
	return unary(a, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) },
		func(x, o float64) float64 { return o * (1 - o) },
		func(x, o float64) float64 { return o * (1 - o) * (1 - 2*o) })
}

func reduceAll(a *Tensor, mean bool) *Tensor {
	n := len(a.Data)
	s := parallelSum(a.Data)
	scale := 1.0
	if mean {
		scale = 1.0 / float64(n)
	}
	res := Scalar(s * scale)
	if recordJets && a.RequiresGrad {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			var d1, d2 float64
			for i := 0; i < n; i++ {
				d1 += a.jet.d[i]
				d2 += a.jet.dd[i]
			}
			res.jet.d[0] = d1 * scale
			res.jet.dd[0] = d2 * scale
		}
	}
	return track1(res, a, func() {
		if !a.RequiresGrad {
			return
		}
		ga := a.ensureGrad()
		g := res.Grad[0] * scale
		parallelFor(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				ga[i] += g
			}
		})
	})
}

func Sum(a *Tensor) *Tensor  { return reduceAll(a, false) }
func Mean(a *Tensor) *Tensor { return reduceAll(a, true) }

func mm(a []float64, m, k int, b []float64, n int) []float64 {
	c := make([]float64, m*n)
	// Rows are independent, so split them across cores for large products.
	runChunks(m, workersFor(m*k*n), func(lo, hi int) {
		for i := lo; i < hi; i++ {
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
	})
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

// mmNT computes a @ wᵀ directly: a is [m,k] row-major, w is [n,k] row-major
// (the [nout, nin] layout dense weights are stored in), and the result is [m,n].
// It reads w in place instead of transposing it first — no [k,n] copy allocated
// and zeroed per call — and both operands are contiguous in the inner product,
// which is what lets this go faster than the general mm.
//
// The inner product runs four independent accumulators over unrolled groups of
// four. A dependent chain of `k` additions stalls on the ~4-cycle latency of the
// FP adder; four chains keep it busy and give a ~1.7x speedup on square inputs.
// This reorders the summation, so the result differs from a naive left-to-right
// sum (and from mm) by a rounding step — within tolerance, not bit-identical.
// Every numeric test that exercises it compares with a tolerance, and no
// byte-exact differential test does a matmul, so the reorder is safe; the
// self-hosted src/tensor.tw stays the naive reference and agrees to tolerance.
func mmNT(a []float64, m, k int, w []float64, n int) []float64 {
	c := make([]float64, m*n)
	// Cache tiling. Without it, the inner j-loop walks all n rows of w for every
	// row of x, so w is re-read from memory m times; once w exceeds the L2 cache
	// (a 2048x2048 weight is 32 MB) the kernel becomes memory-bound and throughput
	// halves. Blocking j into panels that fit L2 keeps each panel resident while
	// the whole x-chunk streams through it, cutting w's memory traffic to roughly
	// once per panel. The block loop is outermost, so for a small n it runs once
	// and this is exactly the untiled kernel.
	//
	// Bit-exact: each c[i,j] is still the full k-length dot summed in k order with
	// the same four accumulators; only the order of visiting (i,j) pairs changes.
	jb := blockNBytes(k, n, 8)
	runChunks(m, workersFor(m*k*n), func(lo, hi int) {
		for j0 := 0; j0 < n; j0 += jb {
			j1 := j0 + jb
			if j1 > n {
				j1 = n
			}
			for i := lo; i < hi; i++ {
				aRow := a[i*k : i*k+k]
				cRow := i * n
				for j := j0; j < j1; j++ {
					wRow := w[j*k : j*k+k]
					var s0, s1, s2, s3 float64
					p := 0
					for ; p+4 <= k; p += 4 {
						s0 += aRow[p] * wRow[p]
						s1 += aRow[p+1] * wRow[p+1]
						s2 += aRow[p+2] * wRow[p+2]
						s3 += aRow[p+3] * wRow[p+3]
					}
					s := (s0 + s1) + (s2 + s3)
					for ; p < k; p++ {
						s += aRow[p] * wRow[p]
					}
					c[cRow+j] = s
				}
			}
		}
	})
	return c
}

// blockNBytes decides the w-row panel width for a weight whose element is
// bytesPer bytes (8 for f64, 1 for int8). Tiling only pays when w is too big to
// stay in cache across the x sweep; below that it just re-reads x for no reason
// (a large-m, small-n product like 4096x4096x256 has a huge x and a w that fits,
// so tiling there is pure loss). So the whole of w fitting a generous last-level
// budget means no tiling (panel = n); past it, panels are sized to L2 so each one
// stays resident while the chunk streams through.
func blockNBytes(k, n, bytesPer int) int {
	if k <= 0 {
		return n
	}
	rowBytes := k * bytesPer
	const llcBudget = 16 * 1024 * 1024 // ~last-level cache; below this, do not tile
	const l2Budget = 512 * 1024        // panel target once tiling is worth it
	if n*rowBytes <= llcBudget {
		return n
	}
	jb := l2Budget / rowBytes
	if jb < 8 {
		jb = 8
	}
	if jb > n {
		jb = n
	}
	return jb
}

// MatMulNT computes a @ bᵀ, where b is the [nout, nin] weight of a dense layer.
// It is the fused form of MatMul(a, Transpose(b)): same result, same gradient,
// but without materialising the transposed weight on every call. a is 1-D [k] or
// 2-D [m,k]; b must be 2-D [n,k]; the result is [n] or [m,n] to match.
func MatMulNT(a, b *Tensor) (*Tensor, error) {
	a2 := a.Shape
	if len(a.Shape) == 1 {
		a2 = []int{1, a.Shape[0]}
	}
	if len(a2) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("linear requires a 1-D or 2-D input and a 2-D weight")
	}
	m, k := a2[0], a2[1]
	n, k2 := b.Shape[0], b.Shape[1]
	if k != k2 {
		return nil, fmt.Errorf("shape mismatch in linear: %v @ %vᵀ (inner %d != %d)", a.Shape, b.Shape, k, k2)
	}
	outData := mmNT(a.Data, m, k, b.Data, n)
	var outShape []int
	if len(a.Shape) == 1 {
		outShape = []int{n}
	} else {
		outShape = []int{m, n}
	}
	res := &Tensor{Data: outData, Shape: outShape}
	if recordJets && (a.RequiresGrad || b.RequiresGrad) {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			// (a@bᵀ)' = a'@bᵀ + a@b'ᵀ; (a@bᵀ)'' = a''@bᵀ + 2 a'@b'ᵀ + a@b''ᵀ.
			ad1 := mmNT(a.jet.d, m, k, b.Data, n)
			bd1 := mmNT(a.Data, m, k, b.jet.d, n)
			ad2 := mmNT(a.jet.dd, m, k, b.Data, n)
			cross := mmNT(a.jet.d, m, k, b.jet.d, n)
			bd2 := mmNT(a.Data, m, k, b.jet.dd, n)
			rd, rdd := res.jet.d, res.jet.dd
			for i := range rd {
				rd[i] = ad1[i] + bd1[i]
				rdd[i] = ad2[i] + 2*cross[i] + bd2[i]
			}
		}
	}
	return track2(res, a, b, func() {
		g := res.Grad // flat length m*n
		if a.RequiresGrad {
			// dA = g @ b, since y = a bᵀ means ∂y[i,j]/∂a[i,p] = b[j,p].
			dA := mm(g, m, n, b.Data, k)
			ga := a.ensureGrad()
			for i := range dA {
				ga[i] += dA[i]
			}
		}
		if b.RequiresGrad {
			// dB = gᵀ @ a, shaped [n,k] like b.
			gt := transpose2d(g, m, n)
			dB := mm(gt, n, m, a.Data, k)
			gb := b.ensureGrad()
			for i := range dB {
				gb[i] += dB[i]
			}
		}
	}), nil
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
	if recordJets && (a.RequiresGrad || b.RequiresGrad) {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			// (a@b)' = a'@b + a@b'; (a@b)'' = a''@b + 2 a'@b' + a@b''.
			ad1 := mm(a.jet.d, m, k, b.Data, n)
			bd1 := mm(a.Data, m, k, b.jet.d, n)
			ad2 := mm(a.jet.dd, m, k, b.Data, n)
			cross := mm(a.jet.d, m, k, b.jet.d, n)
			bd2 := mm(a.Data, m, k, b.jet.dd, n)
			rd, rdd := res.jet.d, res.jet.dd
			for i := range rd {
				rd[i] = ad1[i] + bd1[i]
				rdd[i] = ad2[i] + 2*cross[i] + bd2[i]
			}
		}
	}
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
