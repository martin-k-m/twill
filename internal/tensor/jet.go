package tensor

import "fmt"

// Forward-mode ("jet") automatic differentiation, used for second derivatives.
//
// The reverse-mode engine (Grad) is first-order and not re-differentiable. To
// get exact second derivatives without rewriting it, each supported op also
// wires a `jvp` closure that propagates a first and second directional
// derivative (d, dd) alongside the value. Seeding an input with a direction v
// and walking the graph forward yields the output's directional derivatives; a
// full Hessian follows by polarization over basis directions.

// jetBinary builds the forward-mode closure for an elementwise binary op with
// broadcasting, given the op's first and second partial derivatives.
func jetBinary(res, a, b *Tensor, da, db, daa, dab, dbb func(x, y, o float64) float64) func() {
	shape := res.Shape
	n := len(res.Data)
	rank := len(shape)
	effA := effStrides(a.Shape, shape)
	effB := effStrides(b.Shape, shape)
	return func() {
		coord := make([]int, rank)
		ia, ib := 0, 0
		for o := 0; o < n; o++ {
			x, y, ov := a.Data[ia], b.Data[ib], res.Data[o]
			ax, bx := a.d[ia], b.d[ib]
			res.d[o] = da(x, y, ov)*ax + db(x, y, ov)*bx
			res.dd[o] = da(x, y, ov)*a.dd[ia] + db(x, y, ov)*b.dd[ib] +
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

// directional runs one forward-mode pass over topo, seeding `leaf` with the
// direction v (all other leaves get zero tangent), and returns the output's
// first and second directional derivatives. It errors if the graph uses an op
// without forward-mode support.
func directional(output, leaf *Tensor, topo []*Tensor, v []float64) ([]float64, []float64, error) {
	for _, n := range topo {
		if len(n.d) != len(n.Data) {
			n.d = make([]float64, len(n.Data))
		} else {
			for i := range n.d {
				n.d[i] = 0
			}
		}
		if len(n.dd) != len(n.Data) {
			n.dd = make([]float64, len(n.Data))
		} else {
			for i := range n.dd {
				n.dd[i] = 0
			}
		}
	}
	copy(leaf.d, v)
	for _, n := range topo {
		if n == leaf {
			continue
		}
		if n.jvp != nil {
			n.jvp()
		} else if n.p0 != nil || n.p1 != nil || len(n.pRest) > 0 {
			return nil, nil, fmt.Errorf("second-order autodiff: the function uses an operation without forward-mode support")
		}
	}
	return output.d, output.dd, nil
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
