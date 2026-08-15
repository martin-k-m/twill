// Package ir is twill's intermediate representation: a flat, single-assignment
// dataflow graph sitting between the typed AST and a backend.
//
// Why it exists, and why it looks like this.
//
// docs/CODEGEN.md argues for an IR from the GPU side: you cannot launch one
// kernel for a chain of operations without an object that says what the chain
// is. The profiling from the previous phase says something sharper and closer
// to home. The tree-walking interpreter itself is 0.035% of runtime. About 18%
// goes to allocating and zeroing intermediate buffers that are read once and
// discarded, and about 17% to goroutine coordination in parallelFor. So the
// interpreter is not the thing to remove. The intermediates are.
//
// That measurement decides the shape of the IR:
//
//   - A node names a *value*, not a buffer. Whether a value ever gets memory is
//     decided later, by the fusion pass, and most values in a chain do not.
//     This is the only structural change that can touch the 18%.
//   - The graph is flat and in dependency order, so a greedy single pass can
//     grow fusion regions without a scheduler.
//   - Shapes are concrete integers, not types. twill's checker is best-effort
//     by design (docs/DECISIONS.md entry 3) and cannot type zeros(2, len(xs)).
//     An IR that demanded static shapes would refuse a large fraction of real
//     programs, so shapes here come from concrete values at build time.
//   - Every node carries the op's fusion class, because the class is what the
//     fusion pass and the emitter both switch on, and because a class that
//     lives on the op table cannot drift from the op.
//
// The IR is deliberately narrower than internal/tensor. See Coverage in
// coverage.go for exactly which operators are in and which are not; the
// compiler is not a replacement for the interpreter and does not pretend to be.
package ir

import (
	"fmt"
	"strings"
)

// Op is an IR opcode. The set is a subset of internal/tensor's exported
// operator surface, chosen so that the first backend covers the shape of
// program the fusion pass exists for: an elementwise chain into a reduction,
// and a contraction with an elementwise epilogue.
type Op uint8

const (
	OpInvalid Op = iota

	// Leaves.
	OpParam // a graph input; Attr.Index selects it
	OpConst // a constant buffer; Attr.Index selects it

	// Elementwise, binary, broadcasting.
	OpAdd
	OpSub
	OpMul
	OpDiv
	OpMod
	OpMaximum
	OpMinimum
	OpLt
	OpLe
	OpGt
	OpGe
	OpEq
	OpNe

	// Elementwise, unary.
	OpNeg
	OpExp
	OpLog
	OpSqrt
	OpSin
	OpCos
	OpTanh
	OpSigmoid
	OpRelu
	OpSquare
	OpPowScalar // x ** Attr.F, matching tensor.PowScalar

	// Elementwise, other arities.
	OpWhere // where(cond, a, b), broadcasting all three
	OpClip  // clip(x, lo, hi); bounds in Attr.F

	// Reductions.
	OpSum      // sum over every element, to a scalar
	OpMean     // mean over every element, to a scalar
	OpSumAxis  // sum along Attr.Axis, dropping it
	OpMeanAxis // mean along Attr.Axis, dropping it
	OpSumTo    // sum a broadcast result back to Attr.Shape

	// Structural (pure index arithmetic).
	OpReshape
	OpTranspose   // permutation in Attr.Shape
	OpBroadcastTo // to Attr.Shape

	// Contraction.
	OpMatMul

	opCount
)

// Class is how an op fuses. docs/CODEGEN.md section 3 names three classes and
// the fusion pass in fuse.go is a direct transcription of the rules there.
type Class uint8

const (
	ClassLeaf        Class = iota // param, const: no computation
	ClassElementwise              // one output element per input element tuple
	ClassReduction                // needs a barrier; an elementwise chain fuses into its producer
	ClassStructural               // pure index arithmetic; fuses into a consumer as a remap
	ClassContraction              // its own kernel, with an elementwise epilogue
)

type opInfo struct {
	name  string
	class Class
	arity int // -1 for variadic
}

var opTable = [opCount]opInfo{
	OpInvalid: {"invalid", ClassLeaf, 0},

	OpParam: {"param", ClassLeaf, 0},
	OpConst: {"const", ClassLeaf, 0},

	OpAdd:     {"add", ClassElementwise, 2},
	OpSub:     {"sub", ClassElementwise, 2},
	OpMul:     {"mul", ClassElementwise, 2},
	OpDiv:     {"div", ClassElementwise, 2},
	OpMod:     {"mod", ClassElementwise, 2},
	OpMaximum: {"maximum", ClassElementwise, 2},
	OpMinimum: {"minimum", ClassElementwise, 2},
	OpLt:      {"lt", ClassElementwise, 2},
	OpLe:      {"le", ClassElementwise, 2},
	OpGt:      {"gt", ClassElementwise, 2},
	OpGe:      {"ge", ClassElementwise, 2},
	OpEq:      {"eq", ClassElementwise, 2},
	OpNe:      {"ne", ClassElementwise, 2},

	OpNeg:       {"neg", ClassElementwise, 1},
	OpExp:       {"exp", ClassElementwise, 1},
	OpLog:       {"log", ClassElementwise, 1},
	OpSqrt:      {"sqrt", ClassElementwise, 1},
	OpSin:       {"sin", ClassElementwise, 1},
	OpCos:       {"cos", ClassElementwise, 1},
	OpTanh:      {"tanh", ClassElementwise, 1},
	OpSigmoid:   {"sigmoid", ClassElementwise, 1},
	OpRelu:      {"relu", ClassElementwise, 1},
	OpSquare:    {"square", ClassElementwise, 1},
	OpPowScalar: {"pow_scalar", ClassElementwise, 1},

	OpWhere: {"where", ClassElementwise, 3},
	OpClip:  {"clip", ClassElementwise, 1},

	OpSum:      {"sum", ClassReduction, 1},
	OpMean:     {"mean", ClassReduction, 1},
	OpSumAxis:  {"sum_axis", ClassReduction, 1},
	OpMeanAxis: {"mean_axis", ClassReduction, 1},
	OpSumTo:    {"sum_to", ClassReduction, 1},

	OpReshape:     {"reshape", ClassStructural, 1},
	OpTranspose:   {"transpose", ClassStructural, 1},
	OpBroadcastTo: {"broadcast_to", ClassStructural, 1},

	OpMatMul: {"matmul", ClassContraction, 2},
}

func (o Op) String() string {
	if o >= opCount {
		return fmt.Sprintf("op(%d)", uint8(o))
	}
	return opTable[o].name
}

// Class reports how the op fuses.
func (o Op) Class() Class {
	if o >= opCount {
		return ClassLeaf
	}
	return opTable[o].class
}

// Arity reports the number of operands the op takes.
func (o Op) Arity() int {
	if o >= opCount {
		return 0
	}
	return opTable[o].arity
}

// IsDifferentiable reports whether the op contributes a gradient to its
// operands. The comparisons and the index-producing ops do not: their output is
// piecewise constant in their inputs, so the VJP is structurally zero. This is
// the same convention internal/tensor uses (compareOp builds no backward
// closure).
func (o Op) IsDifferentiable() bool {
	switch o {
	case OpLt, OpLe, OpGt, OpGe, OpEq, OpNe, OpParam, OpConst:
		return false
	}
	return true
}

// Ref names a node by its index in Graph.Nodes. Nodes are in dependency order,
// so a Ref always points backwards and the graph is acyclic by construction.
type Ref int32

// Attr carries an op's small non-tensor parameters. It is one struct rather
// than a per-op union because the set is tiny and a union costs an interface
// allocation per node, which is the thing this whole design is trying to stop
// doing.
type Attr struct {
	Index int     // OpParam, OpConst: which input or constant
	Axis  int     // OpSumAxis, OpMeanAxis
	Shape []int   // OpReshape, OpBroadcastTo, OpSumTo: target; OpTranspose: permutation
	F     float64 // OpClip: lower bound
	G     float64 // OpClip: upper bound
}

// Node is one value in the graph.
type Node struct {
	Op    Op
	In    []Ref
	Shape []int // concrete, always
	Attr  Attr
}

// Graph is a straight-line dataflow graph in dependency order.
//
// Params are named by shape only; the caller supplies the buffers at run time.
// Consts hold their data inline, because a constant folded at trace time (a
// literal, a strike price, a learning rate) has no buffer to be supplied.
type Graph struct {
	Nodes  []Node
	Params []Param
	Consts []Const
	Out    []Ref
}

// Param is a graph input.
type Param struct {
	Name  string
	Shape []int
}

// Const is a constant buffer baked into the graph.
type Const struct {
	Data  []float64
	Shape []int
}

// Numel is the element count of a shape.
func Numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// Strides returns row-major strides, matching strides() in internal/tensor.
func Strides(shape []int) []int {
	s := make([]int, len(shape))
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = acc
		acc *= shape[i]
	}
	return s
}

// EffStrides returns the strides for reading inShape while walking outShape,
// with 0 for a broadcast dimension. This is effStrides() in internal/tensor,
// duplicated rather than exported from there so that the IR package does not
// force a change to the tensor package's surface.
func EffStrides(inShape, outShape []int) []int {
	rIn, rOut := len(inShape), len(outShape)
	real := Strides(inShape)
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

// BroadcastShape is tensor.BroadcastShape, duplicated for the same reason as
// EffStrides.
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

func cloneShape(s []int) []int {
	if s == nil {
		return []int{}
	}
	return append([]int(nil), s...)
}

// ShapeString renders a shape the way twill's diagnostics do.
func ShapeString(s []int) string {
	parts := make([]string, len(s))
	for i, d := range s {
		parts[i] = fmt.Sprint(d)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// String renders the graph one node per line. Used by tests and by
// TWILL_CODEGEN_DUMP.
func (g *Graph) String() string {
	var b strings.Builder
	for i, p := range g.Params {
		fmt.Fprintf(&b, "param %d %s %s\n", i, p.Name, ShapeString(p.Shape))
	}
	for i, c := range g.Consts {
		fmt.Fprintf(&b, "const %d %s (%d elems)\n", i, ShapeString(c.Shape), len(c.Data))
	}
	for i, n := range g.Nodes {
		fmt.Fprintf(&b, "%%%d = %s", i, n.Op)
		switch n.Op {
		case OpParam, OpConst:
			fmt.Fprintf(&b, " %d", n.Attr.Index)
		case OpSumAxis, OpMeanAxis:
			fmt.Fprintf(&b, " axis=%d", n.Attr.Axis)
		case OpReshape, OpBroadcastTo, OpSumTo, OpTranspose:
			fmt.Fprintf(&b, " %s", ShapeString(n.Attr.Shape))
		case OpClip:
			fmt.Fprintf(&b, " lo=%g hi=%g", n.Attr.F, n.Attr.G)
		}
		for _, r := range n.In {
			fmt.Fprintf(&b, " %%%d", r)
		}
		fmt.Fprintf(&b, " : %s\n", ShapeString(n.Shape))
	}
	for _, r := range g.Out {
		fmt.Fprintf(&b, "out %%%d\n", r)
	}
	return b.String()
}
