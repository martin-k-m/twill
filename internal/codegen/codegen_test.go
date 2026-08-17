package codegen_test

import (
	"errors"
	"math"
	"testing"

	"github.com/twill-lang/twill/internal/codegen"
	"github.com/twill-lang/twill/internal/ir"
	"github.com/twill-lang/twill/internal/tensor"
)

func requireCompiler(t *testing.T) {
	t.Helper()
	if _, err := codegen.FindCompiler(); err != nil {
		t.Skipf("%v", err)
	}
}

// requireBackend skips a test that cannot reach the compiled path at all, which
// is a machine without a C compiler or any platform but Windows, where dialling
// into a shared library needs cgo. The tests below assert on what the backend
// computes, so there is nothing to assert when there is no backend.
func requireBackend(t *testing.T) {
	t.Helper()
	b := ir.NewBuilder()
	x := b.Param("x", []int{1})
	b.Output(b.Unary(ir.OpNeg, x))
	g, err := b.Finish()
	if err != nil {
		t.Fatal(err)
	}
	p, err := codegen.Compile(g, codegen.Options{})
	if err != nil {
		if errors.Is(err, codegen.ErrNoCompiler) || errors.Is(err, codegen.ErrNoLoader) {
			t.Skip(err)
		}
		t.Fatal(err)
	}
	p.Close()
}

func compile(t *testing.T, g *ir.Graph, fuse bool) *codegen.Program {
	t.Helper()
	p, err := codegen.Compile(g, codegen.Options{Fuse: fuse})
	if err != nil {
		if errors.Is(err, codegen.ErrNoCompiler) || errors.Is(err, codegen.ErrNoLoader) {
			t.Skip(err)
		}
		t.Fatal(err)
	}
	t.Cleanup(p.Close)
	return p
}

func bitEqual(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("%s: element %d is %v, want %v", name, i, got[i], want[i])
		}
	}
}

// The smallest end-to-end claim: a graph goes through the emitter, a real C
// compiler, and a real dynamic load, and comes back with the interpreter's
// answer to the last bit.
func TestEmittedCodeMatchesInterpreter(t *testing.T) {
	requireCompiler(t)
	b := ir.NewBuilder()
	x := b.Param("x", []int{4, 3})
	y := b.Param("y", []int{3})
	s := b.Binary(ir.OpAdd, b.Binary(ir.OpMul, x, y), b.Scalar(2.5))
	b.Output(b.Unary(ir.OpRelu, b.Binary(ir.OpSub, s, b.Scalar(3))))
	b.Output(b.Sum(s))
	b.Output(b.SumAxis(s, 0))
	b.Output(b.MeanAxis(s, 1))
	g, err := b.Finish()
	if err != nil {
		t.Fatal(err)
	}
	args := []*tensor.Tensor{
		tensor.New([]float64{0.5, -1.2, 0.9, 2.3, -0.7, 1.6, 1.1, -0.6, 0.3, 2.7, -2.2, 0.05}, []int{4, 3}),
		tensor.New([]float64{2.0, -1.0, 0.5}, []int{3}),
	}
	want, err := ir.Eval(g, args)
	if err != nil {
		t.Fatal(err)
	}
	for _, fuse := range []bool{false, true} {
		p := compile(t, g, fuse)
		got, err := p.Run(args)
		if err != nil {
			t.Fatalf("fuse=%v: %v", fuse, err)
		}
		for k := range want {
			bitEqual(t, "output", got[k].Data, want[k].Data)
		}
	}
}

func TestMatMulAndStructural(t *testing.T) {
	requireCompiler(t)
	b := ir.NewBuilder()
	x := b.Param("x", []int{3, 5})
	w := b.Param("w", []int{5, 4})
	bias := b.Param("b", []int{4})
	h := b.Unary(ir.OpTanh, b.Binary(ir.OpAdd, b.MatMul(x, w), bias))
	b.Output(h)
	b.Output(b.Transpose(h, []int{1, 0}))
	b.Output(b.Reshape(h, []int{12}))
	b.Output(b.BroadcastTo(bias, []int{3, 4}))
	g, err := b.Finish()
	if err != nil {
		t.Fatal(err)
	}
	args := []*tensor.Tensor{
		tensor.New(ramp(15, 0.3), []int{3, 5}),
		tensor.New(ramp(20, -0.17), []int{5, 4}),
		tensor.New(ramp(4, 0.9), []int{4}),
	}
	want, err := ir.Eval(g, args)
	if err != nil {
		t.Fatal(err)
	}
	p := compile(t, g, true)
	got, err := p.Run(args)
	if err != nil {
		t.Fatal(err)
	}
	// tanh is libm's here and Go's there, so only the transcendental outputs
	// carry a tolerance. matmul, transpose, reshape and broadcast are exact.
	for k := range want {
		worst := 0.0
		for i := range want[k].Data {
			worst = math.Max(worst, relErr(got[k].Data[i], want[k].Data[i]))
		}
		if worst > 1e-15 {
			t.Fatalf("output %d: worst relative error %g", k, worst)
		}
		t.Logf("output %d: worst relative error %.3g", k, worst)
	}
	bitEqual(t, "broadcast", got[3].Data, want[3].Data)
}

func ramp(n int, step float64) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = math.Sin(float64(i)*1.3+0.2) + step*float64(i)
	}
	return d
}

func relErr(a, b float64) float64 {
	if a == b {
		return 0
	}
	d := math.Abs(a - b)
	m := math.Max(math.Abs(a), math.Abs(b))
	if m < 1e-12 {
		return d
	}
	return d / m
}

// The arena is the measurement. Fusion is supposed to make the compiled program
// allocate less memory than the number of values in the graph implies, and the
// arena size says by how much without any profiling.
func TestFusionShrinksTheArena(t *testing.T) {
	requireCompiler(t)
	const n = 200000
	b := ir.NewBuilder()
	x := b.Param("x", []int{n})
	v := x
	for i := 0; i < 6; i++ {
		v = b.Binary(ir.OpMul, b.Unary(ir.OpExp, b.Binary(ir.OpAdd, v, b.Scalar(0.001))), b.Scalar(0.9))
	}
	b.Output(b.Sum(v))
	g, err := b.Finish()
	if err != nil {
		t.Fatal(err)
	}
	off := compile(t, g, false)
	on := compile(t, g, true)
	t.Logf("arena: %d f64 unfused, %d f64 fused", off.ArenaSize(), on.ArenaSize())
	t.Logf("plan unfused: %s", off.Plan().Stats())
	t.Logf("plan fused:   %s", on.Plan().Stats())
	if on.ArenaSize() >= off.ArenaSize() {
		t.Fatalf("fusion did not shrink the arena")
	}
	args := []*tensor.Tensor{tensor.New(ramp(n, 1e-6), []int{n})}
	a, err := off.Run(args)
	if err != nil {
		t.Fatal(err)
	}
	c, err := on.Run(args)
	if err != nil {
		t.Fatal(err)
	}
	// Fused and unfused must agree bit for bit: fusion changes where values
	// live, never what they are.
	bitEqual(t, "fused vs unfused", c[0].Data, a[0].Data)
}
