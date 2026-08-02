package tensor

import "fmt"

// recordJets controls whether forward ops wire their forward-mode (jvp)
// closures. It is off during normal training and gradient computation — so
// there is zero overhead there — and turned on only while the forward graph for
// a Hessian is being built. The interpreter evaluates sequentially, so a
// package-level flag is safe.
var recordJets bool

// SetRecordJets toggles forward-mode (second-order) recording. Turn it on just
// before building the graph a Hessian will differentiate, and off afterward.
func SetRecordJets(on bool) { recordJets = on }

// Forward-mode ("jet") automatic differentiation, used for second derivatives.
//
// The reverse-mode engine (Grad) is first-order and not re-differentiable. To
// get exact second derivatives without rewriting it, each supported op also
// wires a `jvp` closure that propagates a first and second directional
// derivative (d, dd) alongside the value. Seeding an input with a direction v
// and walking the graph forward yields the output's directional derivatives; a
// full Hessian follows by polarization over basis directions.

// jetBinary builds the forward-mode closure for an elementwise binary op with
// broadcasting, given the op's first and second partial derivatives. The
// broadcast strides are computed lazily inside the closure, so wiring it onto a
// forward op costs nothing until a second derivative is actually taken.
func jetBinary(res, a, b *Tensor, da, db, daa, dab, dbb func(x, y, o float64) float64) func() {
	return func() {
		shape := res.Shape
		n := len(res.Data)
		rank := len(shape)
		effA := effStrides(a.Shape, shape)
		effB := effStrides(b.Shape, shape)
		rd, rdd := res.jet.d, res.jet.dd
		ajd, ajdd, bjd, bjdd := a.jet.d, a.jet.dd, b.jet.d, b.jet.dd
		coord := make([]int, rank)
		ia, ib := 0, 0
		for o := 0; o < n; o++ {
			x, y, ov := a.Data[ia], b.Data[ib], res.Data[o]
			ax, bx := ajd[ia], bjd[ib]
			rd[o] = da(x, y, ov)*ax + db(x, y, ov)*bx
			rdd[o] = da(x, y, ov)*ajdd[ia] + db(x, y, ov)*bjdd[ib] +
				daa(x, y, ov)*ax*ax + 2*dab(x, y, ov)*ax*bx + dbb(x, y, ov)*bx*bx
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
	}
}

// forwardTopo lists the graph rooted at t in forward order (each node after the
// parents it depends on).
func (t *Tensor) forwardTopo() []*Tensor {
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
	return topo
}

// containsTensor reports whether topo holds this exact node. Identity, not
// value: two tensors can carry equal data and still be different graph nodes.
func containsTensor(topo []*Tensor, t *Tensor) bool {
	for _, n := range topo {
		if n == t {
			return true
		}
	}
	return false
}

// directional runs one forward-mode pass over topo, seeding `leaf` with the
// direction v (all other leaves get zero tangent), and returns the output's
// first and second directional derivatives. It errors if the graph uses an op
// without forward-mode support.
//
// The leaf must appear in topo. Callers check that first, because a leaf the
// output does not depend on has no jet state to seed.
func directional(output, leaf *Tensor, topo []*Tensor, v []float64) ([]float64, []float64, error) {
	for _, n := range topo {
		if n.jet == nil {
			n.jet = &jetState{}
		}
		js := n.jet
		if len(js.d) != len(n.Data) {
			js.d = make([]float64, len(n.Data))
		} else {
			for i := range js.d {
				js.d[i] = 0
			}
		}
		if len(js.dd) != len(n.Data) {
			js.dd = make([]float64, len(n.Data))
		} else {
			for i := range js.dd {
				js.dd[i] = 0
			}
		}
	}
	copy(leaf.jet.d, v)
	for _, n := range topo {
		if n == leaf {
			continue
		}
		if n.jet.jvp != nil {
			n.jet.jvp()
		} else if n.p0 != nil || n.p1 != nil || len(n.pRest) > 0 {
			return nil, nil, fmt.Errorf("second-order autodiff: the function uses an operation without forward-mode support")
		}
	}
	return output.jet.d, output.jet.dd, nil
}

// Hessian returns the n×n Hessian of a scalar output with respect to the input
// leaf (n = len(leaf.Data)), computed by forward-mode 2-jets: the diagonal from
// each basis direction's second derivative, and off-diagonals by polarization.
func Hessian(output, leaf *Tensor) ([]float64, int, error) {
	if !output.IsScalar() {
		return nil, 0, fmt.Errorf("hessian: the function must return a scalar")
	}
	n := len(leaf.Data)
	topo := output.forwardTopo()

	// The leaf can be missing from the graph entirely. A forward-only builtin
	// such as floor or a comparison returns an untracked tensor, which severs
	// the parent chain, so walking back from the output never reaches the leaf.
	// The same is true of a function that simply ignores its argument.
	//
	// Zeros rather than an error, for two reasons. It is the right answer: a
	// function the output does not depend on differentiably has an all-zero
	// second derivative, and floor is piecewise constant so its true Hessian is
	// zero everywhere it is defined. And it is the answer the rest of the
	// language already gives, since grad(fn(x) = sum(floor(x))) returns zeros
	// today rather than failing. Erring here would make hessian stricter than
	// grad about the same expression.
	//
	// This also has to come before directional, which seeds the leaf's tangent
	// and would dereference jet state that the init loop never allocated,
	// because that loop only walks topo.
	if !containsTensor(topo, leaf) {
		return make([]float64, n*n), n, nil
	}

	H := make([]float64, n*n)
	diag := make([]float64, n)
	e := make([]float64, n)
	for i := 0; i < n; i++ {
		for k := range e {
			e[k] = 0
		}
		e[i] = 1
		_, dd, err := directional(output, leaf, topo, e)
		if err != nil {
			return nil, 0, err
		}
		diag[i] = dd[0]
		H[i*n+i] = dd[0]
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			for k := range e {
				e[k] = 0
			}
			e[i] = 1
			e[j] = 1
			_, dd, err := directional(output, leaf, topo, e)
			if err != nil {
				return nil, 0, err
			}
			// dd = eᵢ+ⱼᵀ H eᵢ+ⱼ = Hᵢᵢ + 2Hᵢⱼ + Hⱼⱼ.
			h := (dd[0] - diag[i] - diag[j]) / 2
			H[i*n+j] = h
			H[j*n+i] = h
		}
	}
	return H, n, nil
}
