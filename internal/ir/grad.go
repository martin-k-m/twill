package ir

import "fmt"

// Grad transforms a forward graph into a graph that computes both the forward
// outputs and the gradient of every parameter.
//
// docs/CODEGEN.md section 5 states the problem this solves. Today a derivative
// lives in a Go closure created at the moment the operation ran; fuse eight
// operations into one kernel and those eight closures never exist. The answer
// there is to differentiate the *trace* rather than the kernel, and this is
// that transform. Every rule below is a transcription of the corresponding
// da/db/df function in internal/tensor, written as IR rather than as a closure,
// so it is a restatement of code that is already gradient-checked and not new
// mathematics.
//
// The returned graph:
//
//	params  = the original params, then one cotangent per original output
//	outputs = the original outputs, then one gradient per original param
//
// Two consequences worth stating.
//
// The backward work is IR, so it goes through the same fusion pass as the
// forward work. A chain of elementwise VJPs is itself an elementwise chain,
// which is the property that makes the transform worth doing at all: an unfused
// backward pass over a fused forward pass pays back everything the fusion won.
//
// Saved intermediates are explicit. relu's VJP needs its forward input, exp's
// needs its own output, div's needs both operands. Here that is simply a Ref
// into the forward part of the same graph, and whether the value is
// materialised or recomputed is the fusion pass's decision, not this
// transform's. The first implementation lets fusion decide by its ordinary
// use-count rule, which materialises anything read more than once.
func Grad(g *Graph) (*Graph, error) {
	t := &gradT{src: g, b: NewBuilder(), fwd: make([]Ref, len(g.Nodes))}
	return t.run()
}

type gradT struct {
	src *Graph
	b   *Builder
	fwd []Ref   // old node index -> new ref of its forward value
	adj [][]Ref // old node index -> cotangent contributions, in accumulation order
}

func (t *gradT) run() (*Graph, error) {
	// 1. Replay the forward graph into the new builder.
	for i := range t.src.Nodes {
		n := &t.src.Nodes[i]
		in := make([]Ref, len(n.In))
		for k, r := range n.In {
			in[k] = t.fwd[r]
		}
		switch n.Op {
		case OpParam:
			p := t.src.Params[n.Attr.Index]
			t.fwd[i] = t.b.Param(p.Name, p.Shape)
		case OpConst:
			c := t.src.Consts[n.Attr.Index]
			t.fwd[i] = t.b.Const(c.Data, c.Shape)
		default:
			cp := *n
			cp.In = in
			cp.Shape = cloneShape(n.Shape)
			cp.Attr.Shape = cloneShape(n.Attr.Shape)
			t.fwd[i] = t.b.raw(cp)
		}
	}
	for _, r := range t.src.Out {
		t.b.Output(t.fwd[r])
	}

	// 2. Seed the cotangents from fresh parameters, one per output.
	t.adj = make([][]Ref, len(t.src.Nodes))
	for k, r := range t.src.Out {
		ct := t.b.Param(fmt.Sprintf("cotangent%d", k), t.src.Nodes[r].Shape)
		t.adj[r] = append(t.adj[r], ct)
	}

	// 3. Walk backwards. Nodes are in dependency order, so reverse index order
	//    is a valid reverse topological order and every consumer of a node has
	//    already contributed by the time the node is reached. That is the same
	//    order tensor.Backward walks, which is what keeps the accumulation
	//    additions in the same sequence and therefore bit-comparable.
	for i := len(t.src.Nodes) - 1; i >= 0; i-- {
		n := &t.src.Nodes[i]
		if n.Op.Class() == ClassLeaf {
			continue
		}
		gy := t.total(Ref(i))
		if gy < 0 {
			continue // nothing downstream depends on this node
		}
		if !n.Op.IsDifferentiable() {
			continue
		}
		if err := t.vjp(i, n, gy); err != nil {
			return nil, err
		}
	}

	// 4. Emit one gradient per parameter, in parameter order. A parameter that
	//    nothing depends on gets an explicit zero buffer rather than being
	//    omitted, so the caller's indexing never has to know which is which.
	paramNode := make([]int, len(t.src.Params))
	for i := range t.src.Nodes {
		if t.src.Nodes[i].Op == OpParam {
			paramNode[t.src.Nodes[i].Attr.Index] = i
		}
	}
	for pi, ni := range paramNode {
		gr := t.total(Ref(ni))
		if gr < 0 {
			shape := t.src.Params[pi].Shape
			gr = t.b.Const(make([]float64, Numel(shape)), shape)
		}
		t.b.Output(gr)
	}
	return t.b.Finish()
}

// total sums a node's accumulated cotangent contributions left to right,
// returning -1 when there are none.
func (t *gradT) total(r Ref) Ref {
	c := t.adj[r]
	if len(c) == 0 {
		return -1
	}
	acc := c[0]
	for _, x := range c[1:] {
		acc = t.b.Binary(OpAdd, acc, x)
	}
	t.adj[r] = []Ref{acc}
	return acc
}

// accum records a cotangent contribution for an operand, reducing it back to
// the operand's own shape when the forward op broadcast it.
func (t *gradT) accum(old Ref, contrib Ref) {
	want := t.src.Nodes[old].Shape
	if !shapeEqual(t.b.Shape(contrib), want) {
		contrib = t.b.SumTo(contrib, want)
	}
	t.adj[old] = append(t.adj[old], contrib)
}

func (t *gradT) c(x float64) Ref { return t.b.Scalar(x) }

// vjp emits the vector-Jacobian product for one node.
func (t *gradT) vjp(i int, n *Node, gy Ref) error {
	B := t.b
	// a, b, c are the operands' forward refs; y is this node's forward ref.
	var a, bb, cc Ref
	if len(n.In) > 0 {
		a = t.fwd[n.In[0]]
	}
	if len(n.In) > 1 {
		bb = t.fwd[n.In[1]]
	}
	if len(n.In) > 2 {
		cc = t.fwd[n.In[2]]
	}
	y := t.fwd[i]
	inShape := func(k int) []int { return t.src.Nodes[n.In[k]].Shape }

	switch n.Op {
	case OpAdd:
		t.accum(n.In[0], gy)
		t.accum(n.In[1], gy)
	case OpSub:
		t.accum(n.In[0], gy)
		t.accum(n.In[1], B.Unary(OpNeg, gy))
	case OpMul:
		t.accum(n.In[0], B.Binary(OpMul, bb, gy))
		t.accum(n.In[1], B.Binary(OpMul, a, gy))
	case OpDiv:
		// tensor.Div's da is 1/y and db is -x/(y*y), each multiplied by the
		// cotangent in that order. Written the same way here so the rounding is
		// the same rounding.
		t.accum(n.In[0], B.Binary(OpMul, B.Binary(OpDiv, t.c(1), bb), gy))
		num := B.Unary(OpNeg, a)
		den := B.Binary(OpMul, bb, bb)
		t.accum(n.In[1], B.Binary(OpMul, B.Binary(OpDiv, num, den), gy))
	case OpMod:
		// da is 1; db is -floor(x/y), and floor is not an IR op. Rather than
		// add an op for a case no twill program differentiates, refuse.
		t.accum(n.In[0], gy)
		if t.dependsOnParam(n.In[1]) {
			return fmt.Errorf("gradient through the divisor of %% is not supported by the compiler")
		}
	case OpMaximum:
		t.accum(n.In[0], B.Binary(OpMul, B.Binary(OpGe, a, bb), gy))
		t.accum(n.In[1], B.Binary(OpMul, B.Binary(OpLt, a, bb), gy))
	case OpMinimum:
		t.accum(n.In[0], B.Binary(OpMul, B.Binary(OpLe, a, bb), gy))
		t.accum(n.In[1], B.Binary(OpMul, B.Binary(OpGt, a, bb), gy))

	case OpNeg:
		t.accum(n.In[0], B.Unary(OpNeg, gy))
	case OpExp:
		t.accum(n.In[0], B.Binary(OpMul, y, gy))
	case OpLog:
		t.accum(n.In[0], B.Binary(OpMul, B.Binary(OpDiv, t.c(1), a), gy))
	case OpSqrt:
		t.accum(n.In[0], B.Binary(OpMul, B.Binary(OpDiv, t.c(0.5), y), gy))
	case OpSin:
		t.accum(n.In[0], B.Binary(OpMul, B.Unary(OpCos, a), gy))
	case OpCos:
		t.accum(n.In[0], B.Binary(OpMul, B.Unary(OpNeg, B.Unary(OpSin, a)), gy))
	case OpTanh:
		d := B.Binary(OpSub, t.c(1), B.Binary(OpMul, y, y))
		t.accum(n.In[0], B.Binary(OpMul, d, gy))
	case OpSigmoid:
		d := B.Binary(OpMul, y, B.Binary(OpSub, t.c(1), y))
		t.accum(n.In[0], B.Binary(OpMul, d, gy))
	case OpRelu:
		t.accum(n.In[0], B.Binary(OpMul, B.Binary(OpGt, a, t.c(0)), gy))
	case OpSquare:
		t.accum(n.In[0], B.Binary(OpMul, B.Binary(OpMul, t.c(2), a), gy))
	case OpPowScalar:
		p := n.Attr.F
		d := B.Binary(OpMul, t.c(p), B.PowScalar(a, p-1))
		t.accum(n.In[0], B.Binary(OpMul, d, gy))
	case OpClip:
		lo := B.Binary(OpGt, a, t.c(n.Attr.F))
		hi := B.Binary(OpLt, a, t.c(n.Attr.G))
		t.accum(n.In[0], B.Binary(OpMul, B.Binary(OpMul, lo, hi), gy))
	case OpWhere:
		zero := t.c(0)
		t.accum(n.In[1], B.Where(a, gy, zero))
		t.accum(n.In[2], B.Where(a, zero, gy))
		_ = cc

	case OpSum:
		t.accum(n.In[0], B.BroadcastTo(gy, inShape(0)))
	case OpMean:
		scaled := B.Binary(OpMul, gy, t.c(1.0/float64(Numel(inShape(0)))))
		t.accum(n.In[0], B.BroadcastTo(scaled, inShape(0)))
	case OpSumAxis:
		t.accum(n.In[0], t.expandAxis(gy, inShape(0), n.Attr.Axis))
	case OpMeanAxis:
		L := inShape(0)[n.Attr.Axis]
		scaled := B.Binary(OpMul, gy, t.c(1.0/float64(L)))
		t.accum(n.In[0], t.expandAxis(scaled, inShape(0), n.Attr.Axis))
	case OpSumTo:
		t.accum(n.In[0], B.BroadcastTo(gy, inShape(0)))

	case OpReshape:
		t.accum(n.In[0], B.Reshape(gy, inShape(0)))
	case OpTranspose:
		perm := n.Attr.Shape
		inv := make([]int, len(perm))
		for i, p := range perm {
			inv[p] = i
		}
		t.accum(n.In[0], B.Transpose(gy, inv))
	case OpBroadcastTo:
		t.accum(n.In[0], B.SumTo(gy, inShape(0)))

	case OpMatMul:
		// tensor.MatMul's closure is dA = g @ bᵀ and dB = aᵀ @ g on the 2-D
		// promotions of the operands, so the transform promotes the same way and
		// reshapes the answer back.
		as, bs := inShape(0), inShape(1)
		a2, b2 := as, bs
		if len(as) == 1 {
			a2 = []int{1, as[0]}
		}
		if len(bs) == 1 {
			b2 = []int{bs[0], 1}
		}
		m, k, nn := a2[0], a2[1], b2[1]
		a2d, b2d := a, bb
		if len(as) == 1 {
			a2d = B.Reshape(a, a2)
		}
		if len(bs) == 1 {
			b2d = B.Reshape(bb, b2)
		}
		g2d := gy
		if !shapeEqual(t.b.Shape(gy), []int{m, nn}) {
			g2d = B.Reshape(gy, []int{m, nn})
		}
		dA := B.MatMul(g2d, B.Transpose(b2d, []int{1, 0}))
		dB := B.MatMul(B.Transpose(a2d, []int{1, 0}), g2d)
		if len(as) == 1 {
			dA = B.Reshape(dA, as)
		}
		if len(bs) == 1 {
			dB = B.Reshape(dB, bs)
		}
		_ = k
		t.accum(n.In[0], dA)
		t.accum(n.In[1], dB)

	default:
		return fmt.Errorf("no VJP rule for %s", n.Op)
	}
	return B.Err()
}

// expandAxis undoes an axis reduction: reinsert the reduced axis with extent 1
// and broadcast it back out.
func (t *gradT) expandAxis(gy Ref, inShape []int, axis int) Ref {
	keep := make([]int, 0, len(inShape))
	keep = append(keep, inShape[:axis]...)
	keep = append(keep, 1)
	keep = append(keep, inShape[axis+1:]...)
	return t.b.BroadcastTo(t.b.Reshape(gy, keep), inShape)
}

// dependsOnParam reports whether a node's value is a function of any graph
// parameter, which is what decides whether a missing VJP rule actually matters.
func (t *gradT) dependsOnParam(r Ref) bool {
	seen := make([]bool, len(t.src.Nodes))
	var walk func(Ref) bool
	walk = func(x Ref) bool {
		if seen[x] {
			return false
		}
		seen[x] = true
		n := &t.src.Nodes[x]
		if n.Op == OpParam {
			return true
		}
		for _, in := range n.In {
			if walk(in) {
				return true
			}
		}
		return false
	}
	return walk(r)
}
