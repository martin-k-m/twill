package codegen

import (
	"fmt"
	"sync"

	"github.com/martin-k-m/twill/internal/ir"
	"github.com/martin-k-m/twill/internal/tensor"
)

// Program is a compiled graph: emitted C, built into a shared library, loaded,
// and ready to be called on a set of inputs.
type Program struct {
	plan  *ir.Plan
	lay   Layout
	src   string
	lib   string
	e     *entry
	mu    sync.Mutex
	arena []float64
}

// Options control compilation.
type Options struct {
	// Fuse chooses the fusion setting. The differential harness runs both, and
	// they must agree, which is the property that says fusion is a performance
	// change and not a semantic one.
	Fuse bool
}

// Compile turns a graph into a running program.
func Compile(g *ir.Graph, opt Options) (*Program, error) {
	plan := ir.FuseOff(g)
	if opt.Fuse {
		plan = ir.Fuse(g)
	}
	src, lay, err := Emit(plan)
	if err != nil {
		return nil, err
	}
	lib, err := buildShared(src)
	if err != nil {
		return nil, err
	}
	e, err := loadEntry(lib)
	if err != nil {
		return nil, err
	}
	p := &Program{plan: plan, lay: lay, src: src, lib: lib, e: e, arena: make([]float64, lay.Total)}
	// Constants are written once. Nothing in the emitted code writes to a
	// constant's offset, because only a region root is ever a destination and a
	// constant's region computes nothing.
	for i, n := range g.Nodes {
		if n.Op == ir.OpConst {
			copy(p.arena[lay.Offset[i]:], g.Consts[n.Attr.Index].Data)
		}
	}
	return p, nil
}

// Source returns the emitted C. Tests read it, and so should anyone deciding
// whether to believe the backend.
func (p *Program) Source() string { return p.src }

// Library returns the path of the compiled shared object.
func (p *Program) Library() string { return p.lib }

// Plan returns the fusion plan the program was compiled from.
func (p *Program) Plan() *ir.Plan { return p.plan }

// ArenaSize is the total f64 elements the program allocates, which is the
// direct measure of what the fusion pass saved.
func (p *Program) ArenaSize() int { return p.lay.Total }

// Close unloads the library.
func (p *Program) Close() { p.e.close() }

// Run evaluates the program on the given arguments.
func (p *Program) Run(args []*tensor.Tensor) ([]*tensor.Tensor, error) {
	g := p.plan.Graph
	if len(args) != len(g.Params) {
		return nil, fmt.Errorf("codegen: graph takes %d params, given %d", len(g.Params), len(args))
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for i, n := range g.Nodes {
		if n.Op != ir.OpParam {
			continue
		}
		a := args[n.Attr.Index]
		if !shapeEq(a.Shape, n.Shape) {
			return nil, fmt.Errorf("codegen: param %d (%s): shape %s, want %s",
				n.Attr.Index, g.Params[n.Attr.Index].Name,
				ir.ShapeString(a.Shape), ir.ShapeString(n.Shape))
		}
		if a.DType() != tensor.DTF64 {
			return nil, fmt.Errorf("codegen: param %d is %v; the backend is f64 only",
				n.Attr.Index, a.DType())
		}
		copy(p.arena[p.lay.Offset[i]:], a.Data)
	}
	if err := p.e.call(p.arena); err != nil {
		return nil, err
	}
	outs := make([]*tensor.Tensor, len(g.Out))
	for k, r := range g.Out {
		n := &g.Nodes[r]
		off := p.lay.Offset[r]
		if off < 0 {
			return nil, fmt.Errorf("codegen: output %%%d was not materialised", r)
		}
		data := make([]float64, ir.Numel(n.Shape))
		copy(data, p.arena[off:])
		outs[k] = tensor.New(data, append([]int(nil), n.Shape...))
	}
	return outs, nil
}
