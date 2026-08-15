package ir

import (
	"fmt"

	"github.com/martin-k-m/twill/internal/tensor"
)

// Eval runs a graph by dispatching every node to the internal/tensor function
// the interpreter would have called.
//
// This is the reference semantics for the IR, and it is deliberately not a
// second implementation of anything. docs/design.md says the interpreter is the
// reference, and the interpreter is a thin layer over internal/tensor; so an
// evaluator that calls the same functions in the same order agrees with the
// interpreter by construction, and any disagreement between it and the
// interpreter is a defect in the *tracer*, not in arithmetic. That separation is
// what makes the differential test in stage 3 able to say which layer is wrong.
//
// It is also the fallback: a graph whose ops the backend cannot emit still runs
// here.
func Eval(g *Graph, args []*tensor.Tensor) ([]*tensor.Tensor, error) {
	vals, err := evalNodes(g, args)
	if err != nil {
		return nil, err
	}
	out := make([]*tensor.Tensor, len(g.Out))
	for i, r := range g.Out {
		out[i] = vals[r]
	}
	return out, nil
}

func evalNodes(g *Graph, args []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if len(args) != len(g.Params) {
		return nil, fmt.Errorf("graph takes %d params, given %d", len(g.Params), len(args))
	}
	for i, a := range args {
		if !shapeEqual(a.Shape, g.Params[i].Shape) {
			return nil, fmt.Errorf("param %d (%s): shape %s, want %s",
				i, g.Params[i].Name, ShapeString(a.Shape), ShapeString(g.Params[i].Shape))
		}
		if a.DType() != tensor.DTF64 {
			return nil, fmt.Errorf("param %d (%s): dtype %v, the IR is f64 only",
				i, g.Params[i].Name, a.DType())
		}
	}
	vals := make([]*tensor.Tensor, len(g.Nodes))
	for i := range g.Nodes {
		n := &g.Nodes[i]
		v, err := evalNode(g, n, vals)
		if err != nil {
			return nil, fmt.Errorf("%%%d (%s): %w", i, n.Op, err)
		}
		if n.Op == OpParam {
			v = args[n.Attr.Index]
		}
		if !shapeEqual(v.Shape, n.Shape) {
			return nil, fmt.Errorf("%%%d (%s): produced %s, IR says %s",
				i, n.Op, ShapeString(v.Shape), ShapeString(n.Shape))
		}
		vals[i] = v
	}
	return vals, nil
}

func evalNode(g *Graph, n *Node, vals []*tensor.Tensor) (*tensor.Tensor, error) {
	in := func(k int) *tensor.Tensor { return vals[n.In[k]] }
	switch n.Op {
	case OpParam:
		return tensor.New(nil, cloneShape(n.Shape)), nil // replaced by the caller
	case OpConst:
		c := g.Consts[n.Attr.Index]
		return tensor.New(append([]float64(nil), c.Data...), cloneShape(c.Shape)), nil

	case OpAdd:
		return tensor.Add(in(0), in(1))
	case OpSub:
		return tensor.Sub(in(0), in(1))
	case OpMul:
		return tensor.Mul(in(0), in(1))
	case OpDiv:
		return tensor.Div(in(0), in(1))
	case OpMod:
		return tensor.Mod(in(0), in(1))
	case OpMaximum:
		return tensor.Maximum(in(0), in(1))
	case OpMinimum:
		return tensor.Minimum(in(0), in(1))
	case OpLt:
		return tensor.Less(in(0), in(1))
	case OpLe:
		return tensor.LessEqual(in(0), in(1))
	case OpGt:
		return tensor.Greater(in(0), in(1))
	case OpGe:
		return tensor.GreaterEqual(in(0), in(1))
	case OpEq:
		return tensor.EqualOp(in(0), in(1))
	case OpNe:
		return tensor.NotEqual(in(0), in(1))

	case OpNeg:
		return tensor.Neg(in(0)), nil
	case OpExp:
		return tensor.Exp(in(0)), nil
	case OpLog:
		return tensor.Log(in(0)), nil
	case OpSqrt:
		return tensor.Sqrt(in(0)), nil
	case OpSin:
		return tensor.Sin(in(0)), nil
	case OpCos:
		return tensor.Cos(in(0)), nil
	case OpTanh:
		return tensor.Tanh(in(0)), nil
	case OpSigmoid:
		return tensor.Sigmoid(in(0)), nil
	case OpRelu:
		return tensor.Relu(in(0)), nil
	case OpSquare:
		return tensor.Square(in(0)), nil
	case OpPowScalar:
		return tensor.PowScalar(in(0), n.Attr.F), nil
	case OpClip:
		return tensor.Clip(in(0), n.Attr.F, n.Attr.G), nil
	case OpWhere:
		return tensor.Where(in(0), in(1), in(2))

	case OpSum:
		return tensor.Sum(in(0)), nil
	case OpMean:
		return tensor.Mean(in(0)), nil
	case OpSumAxis:
		return tensor.SumAxis(in(0), n.Attr.Axis)
	case OpMeanAxis:
		return tensor.MeanAxis(in(0), n.Attr.Axis)
	case OpSumTo:
		return tensor.SumTo(in(0), n.Attr.Shape)

	case OpReshape:
		return tensor.Reshape(in(0), n.Attr.Shape)
	case OpTranspose:
		return tensor.TransposePerm(in(0), n.Attr.Shape)
	case OpBroadcastTo:
		return tensor.BroadcastTo(in(0), n.Attr.Shape)

	case OpMatMul:
		return tensor.MatMul(in(0), in(1))
	}
	return nil, fmt.Errorf("no reference evaluation for %s", n.Op)
}

// EvalWithGrad runs the graph on gradient-tracking leaves, backpropagates the
// given cotangents into the outputs, and returns both the outputs and the
// gradient of each parameter.
//
// The gradients come from tensor.Backward, that is, from the interpreter's own
// autodiff. This is the oracle stage 3 compares the *transformed* backward
// graph (grad.go) against, and it is a sharper oracle than finite differences
// because it agrees to the last bit rather than to 1e-7.
func EvalWithGrad(g *Graph, args []*tensor.Tensor, cotangents []*tensor.Tensor) (outs []*tensor.Tensor, grads [][]float64, err error) {
	leaves := make([]*tensor.Tensor, len(args))
	for i, a := range args {
		leaves[i] = tensor.Leaf(a.Data, a.Shape)
	}
	vals, err := evalNodes(g, leaves)
	if err != nil {
		return nil, nil, err
	}
	outs = make([]*tensor.Tensor, len(g.Out))
	for i, r := range g.Out {
		outs[i] = vals[r]
	}
	if len(cotangents) != len(outs) {
		return nil, nil, fmt.Errorf("%d cotangents for %d outputs", len(cotangents), len(outs))
	}
	// Reduce the outputs to one scalar with the supplied cotangents and
	// differentiate that, which is the same reduction fullgradcheck_test.go uses
	// and for the same reason: a single scalar makes one backward pass enough,
	// and an irregular cotangent makes index-shuffling bugs visible.
	var loss *tensor.Tensor
	for i, o := range outs {
		p, err := tensor.Mul(o, cotangents[i])
		if err != nil {
			return nil, nil, err
		}
		s := tensor.Sum(p)
		if loss == nil {
			loss = s
		} else {
			loss, err = tensor.Add(loss, s)
			if err != nil {
				return nil, nil, err
			}
		}
	}
	if loss == nil {
		return outs, make([][]float64, 0), nil
	}
	if err := loss.Backward(); err != nil {
		return nil, nil, err
	}
	grads = make([][]float64, len(leaves))
	for i, l := range leaves {
		gr := l.Grad
		if gr == nil {
			gr = make([]float64, len(l.Data))
		}
		grads[i] = append([]float64(nil), gr...)
	}
	return outs, grads, nil
}
