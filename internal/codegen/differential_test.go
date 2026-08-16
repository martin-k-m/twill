package codegen_test

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/martin-k-m/twill/internal/codegen"
	"github.com/martin-k-m/twill/internal/ir"
	"github.com/martin-k-m/twill/internal/tensor"
)

// Differential testing of the compiler against the interpreter.
//
// docs/design.md makes the interpreter the reference semantics, so the
// compiler's correctness is a testable proposition rather than a review
// criterion: the compiler is correct when it agrees with the interpreter.
// docs/CODEGEN.md section 7 sets the bar, and this file is that bar enforced.
//
// # The two tolerance classes, and why the line falls where it does
//
// BIT-EXACT. Fusion changes where a value lives, never what it is. An
// elementwise chain computes exactly the same IEEE operations on exactly the
// same values in exactly the same order whether or not the intermediates are
// materialised, so `exp(x) * y` fused must equal `exp(x) * y` unfused to the
// last bit and anything else is a defect. The same holds for the operations
// whose C spelling is the same IEEE operation as Go's:
//
//	+  -  *  /  sqrt  square  neg  relu  clip  where  min  max
//	the six comparisons
//	sum  mean  sum_axis  mean_axis  sum_to
//	reshape  transpose  broadcast_to
//	matmul
//
// sqrt is on that list because IEEE 754 requires it to be correctly rounded, so
// Go's and libm's cannot differ. The reductions are on it because the emitted
// code carries internal/tensor's accumulation order, including parallelSum's
// fixed 4096-element blocking, and because the build passes -ffp-contract=off
// so no multiply-add fuses. matmul is on it because the emitted loop is mm()
// transcribed, zero skip included.
//
// TOLERANCE-BOUNDED. IEEE 754 does not require a correctly rounded exp, log,
// sin, cos, tanh or pow, and Go implements its own rather than calling libm.
// So these differ from the C compiler's in the last bits and there is no
// arrangement of the emitter that would fix it:
//
//	exp  log  sin  cos  tanh  sigmoid  pow_scalar
//
// The tolerance is stated in two parts, because a single number would either be
// too loose to catch a defect or too tight to be true.
//
// At the primitive, the bound is 1e-15 relative, and it is measured rather than
// assumed: TestTranscendentalULP runs each function over its useful range
// through Go and through the emitted code and reports the disagreement. On the
// machine in the report it is at most 1 unit in the last place, worst relative
// 2.22e-16, for every one of them, and sqrt does not differ at all. The bound
// sits above that measurement with room rather than being fitted to it, and the
// measurement test fails if a toolchain ever narrows the gap.
//
// At the whole program, the bound is 1e-12 relative, and the extra four orders
// are not slack in the backend. They are conditioning. The generator freely
// writes things like cos(x) - x, and a subtraction of two nearby quantities
// multiplies the relative error of its operands by the ratio of their magnitude
// to the result's. One such expression turned the 1-ulp cos difference above
// into 1.48e-15 on the output, which is a correct backend meeting an
// ill-conditioned program. Rather than hide that by comparing only
// well-conditioned programs, the corpus keeps them and the bound admits the
// amplification; the tests report the worst disagreement they actually saw, so
// the headroom between the bound and the measurement is visible instead of
// merely claimed.
//
// The one comparison that stays exact for every program, transcendental or not,
// is fused against unfused: both go through the same libm, so any difference at
// all between them is a fusion defect. That assertion is in
// TestDifferentialForward and it is the sharpest thing in this file.
//
// GRADIENTS. Three oracles, and the sharpest one is not the one that was
// expected.
//
// The plan was to compare the compiled backward pass against tensor.Backward
// bit for bit, on the grounds that both compute the same VJPs on the same
// values. That is true for a value with one consumer and false for a value with
// several, and the reason is worth writing down because it is a property of
// reverse mode and not a defect in either side. When a value feeds two
// operations, its cotangent is a sum of two contributions, and floating-point
// addition is not associative. tensor.Backward adds them in the order its
// depth-first topological walk happens to visit the consumers; the transform in
// ir/grad.go adds them in reverse node-index order. Both orders are correct and
// they differ in the last bits. The measured disagreement is at most 5.4e-16 on
// the corpus below, one part in 4.5 quadrillion, and it appears only on
// parameters with more than one consumer.
//
// So the three oracles are:
//
//	compiled fused vs compiled unfused   bit-exact, asserted, no exceptions
//	compiled vs tensor.Backward          1e-12 relative, for the reason above
//	compiled vs finite differences       1e-7 relative, the method and the
//	                                     tolerance of fullgradcheck_test.go
//
// The first is the sharp one: it is the compiler compared against itself with
// the only variable being the thing under test. The third is the one that would
// survive the interpreter itself being wrong, which is the gap docs/CODEGEN.md
// section 7.4 names.

const (
	// bitExact is the bar for anything that does not reassociate arithmetic.
	bitExact = 0.0
	// transTol is the bar for a whole generated program containing a
	// transcendental, and it covers the conditioning of the program as well as
	// the primitive. See above.
	transTol = 1e-12
	// ulpRel is the bar for a bare transcendental, one call, no cancellation.
	ulpRel = 1e-15
	// gradTol is the bar against tensor.Backward. See the note on accumulation
	// order above: it is not slack in the backend.
	gradTol = 1e-12
)

// --- the program generator -------------------------------------------------

// genShapes is deliberately small and includes 1 and rank 0. A dimension of 1
// broadcasts against anything and is the case a naive equality rule gets wrong,
// and a rank-0 operand is what most scalar constants in a real program are.
var genShapes = [][]int{
	{}, {4}, {6}, {3, 4}, {1, 4}, {3, 1}, {2, 3, 4},
}

type gen struct {
	b       *ir.Builder
	r       *rand.Rand
	live    []ir.Ref
	inexact bool // the graph contains an operation Go and libm may round differently
}

func (g *gen) shape(r ir.Ref) []int { return g.b.Shape(r) }

// pick returns a live value, preferring one whose shape broadcasts with want.
func (g *gen) pick() ir.Ref { return g.live[g.r.Intn(len(g.live))] }

func (g *gen) pickCompatible(with ir.Ref) (ir.Ref, bool) {
	order := g.r.Perm(len(g.live))
	for _, i := range order {
		if _, ok := ir.BroadcastShape(g.shape(with), g.shape(g.live[i])); ok {
			return g.live[i], true
		}
	}
	return 0, false
}

// positive wraps a value so that log and sqrt see a strictly positive operand,
// which is what keeps a generated program from being a NaN factory rather than
// a test.
func (g *gen) positive(x ir.Ref) ir.Ref {
	return g.b.Binary(ir.OpAdd, g.b.Unary(ir.OpSquare, x), g.b.Scalar(0.5))
}

// small keeps exp's argument in a range where the result is finite and the
// relative comparison is meaningful.
func (g *gen) small(x ir.Ref) ir.Ref {
	return g.b.Binary(ir.OpMul, g.b.Unary(ir.OpTanh, x), g.b.Scalar(2))
}

func (g *gen) step() {
	b := g.b
	switch g.r.Intn(18) {
	case 0, 1, 2, 3: // elementwise binary
		x := g.pick()
		y, ok := g.pickCompatible(x)
		if !ok {
			return
		}
		op := []ir.Op{ir.OpAdd, ir.OpSub, ir.OpMul, ir.OpMaximum, ir.OpMinimum}[g.r.Intn(5)]
		g.live = append(g.live, b.Binary(op, x, y))
	case 4: // division, with a divisor that cannot be zero
		x := g.pick()
		y, ok := g.pickCompatible(x)
		if !ok {
			return
		}
		g.live = append(g.live, b.Binary(ir.OpDiv, x, g.positive(y)))
	case 5, 6: // exact unary
		op := []ir.Op{ir.OpNeg, ir.OpSquare, ir.OpRelu}[g.r.Intn(3)]
		g.live = append(g.live, b.Unary(op, g.pick()))
	case 7:
		g.live = append(g.live, b.Clip(g.pick(), -1.25, 1.75))
	case 8:
		g.live = append(g.live, b.Unary(ir.OpSqrt, g.positive(g.pick())))
	case 9, 10: // transcendental unary
		g.inexact = true
		op := []ir.Op{ir.OpExp, ir.OpSin, ir.OpCos, ir.OpTanh, ir.OpSigmoid}[g.r.Intn(5)]
		x := g.pick()
		if op == ir.OpExp {
			x = g.small(x)
		}
		g.live = append(g.live, b.Unary(op, x))
	case 11:
		g.inexact = true
		g.live = append(g.live, b.Unary(ir.OpLog, g.positive(g.pick())))
	case 12: // where, driven by a comparison so the mask is a real mask
		x := g.pick()
		y, ok := g.pickCompatible(x)
		if !ok {
			return
		}
		cond := b.Binary(ir.OpGt, x, y)
		g.live = append(g.live, b.Where(cond, x, y))
	case 13: // full reduction
		x := g.pick()
		if g.r.Intn(2) == 0 {
			g.live = append(g.live, b.Sum(x))
		} else {
			g.live = append(g.live, b.Mean(x))
		}
	case 14: // axis reduction
		x := g.pick()
		if len(g.shape(x)) == 0 {
			return
		}
		axis := g.r.Intn(len(g.shape(x)))
		if g.r.Intn(2) == 0 {
			g.live = append(g.live, b.SumAxis(x, axis))
		} else {
			g.live = append(g.live, b.MeanAxis(x, axis))
		}
	case 15: // reshape
		x := g.pick()
		n := ir.Numel(g.shape(x))
		var cands [][]int
		for _, s := range genShapes {
			if ir.Numel(s) == n {
				cands = append(cands, s)
			}
		}
		if len(cands) == 0 {
			return
		}
		g.live = append(g.live, b.Reshape(x, cands[g.r.Intn(len(cands))]))
	case 16: // transpose
		x := g.pick()
		if len(g.shape(x)) < 2 {
			return
		}
		perm := g.r.Perm(len(g.shape(x)))
		g.live = append(g.live, b.Transpose(x, perm))
	case 17: // broadcast_to
		x := g.pick()
		for _, s := range genShapes {
			if up, ok := ir.BroadcastShape(g.shape(x), s); ok && sameShape(up, s) && !sameShape(g.shape(x), s) {
				g.live = append(g.live, b.BroadcastTo(x, s))
				return
			}
		}
	}
}

type program struct {
	g       *ir.Graph
	args    []*tensor.Tensor
	inexact bool
}

// generate builds one random program. Every parameter gets deterministic data
// from the seed, so a failure reproduces exactly.
func generate(r *rand.Rand, depth int) (*program, error) {
	b := ir.NewBuilder()
	g := &gen{b: b, r: r}
	nparam := 2 + r.Intn(2)
	var args []*tensor.Tensor
	for i := 0; i < nparam; i++ {
		s := genShapes[r.Intn(len(genShapes))]
		g.live = append(g.live, b.Param(fmt.Sprintf("p%d", i), s))
		d := make([]float64, ir.Numel(s))
		for j := range d {
			d[j] = math.Sin(float64(i)*3.1+float64(j)*1.37+0.4)*1.8 + 0.15
		}
		args = append(args, tensor.New(d, s))
	}
	// A matmul in about a third of the programs, on operands built for it, since
	// a random pair of live shapes almost never contracts.
	if r.Intn(3) == 0 {
		m, k, n := 1+r.Intn(3), 1+r.Intn(4), 1+r.Intn(3)
		x := b.Param("mmA", []int{m, k})
		y := b.Param("mmB", []int{k, n})
		args = append(args,
			tensor.New(rampN(m*k, 0.21), []int{m, k}),
			tensor.New(rampN(k*n, -0.13), []int{k, n}))
		g.live = append(g.live, b.MatMul(x, y))
	}
	for i := 0; i < depth; i++ {
		g.step()
		if b.Err() != nil {
			return nil, b.Err()
		}
	}
	// One to three outputs, always including the last value produced so the
	// deepest part of the graph is never dead.
	outs := map[ir.Ref]bool{g.live[len(g.live)-1]: true}
	for i := 0; i < r.Intn(3); i++ {
		outs[g.live[r.Intn(len(g.live))]] = true
	}
	keys := make([]int, 0, len(outs))
	for k := range outs {
		keys = append(keys, int(k))
	}
	sort.Ints(keys)
	for _, k := range keys {
		b.Output(ir.Ref(k))
	}
	graph, err := b.Finish()
	if err != nil {
		return nil, err
	}
	return &program{g: graph, args: args, inexact: g.inexact}, nil
}

func sameShape(a, b []int) bool {
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

func rampN(n int, step float64) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = math.Cos(float64(i)*0.83+0.1) + step*float64(i)
	}
	return d
}

func finite(ts []*tensor.Tensor) bool {
	for _, t := range ts {
		for _, v := range t.Data {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e12 {
				return false
			}
		}
	}
	return true
}

// worstRel reports the largest relative disagreement, treating a pair of
// identical bit patterns as zero so that signed zeros and equal infinities do
// not register.
func worstRel(a, b []float64) (float64, int) {
	worst, at := 0.0, -1
	for i := range a {
		if math.Float64bits(a[i]) == math.Float64bits(b[i]) {
			continue
		}
		e := relErr(a[i], b[i])
		if e > worst {
			worst, at = e, i
		}
	}
	return worst, at
}

// --- the tests -------------------------------------------------------------

// nPrograms is the size of the generated corpus. It is the number quoted in the
// report, so it lives here rather than being spelled inline twice.
//
// Each program is compiled twice by a real C compiler, so the full corpus costs
// a few minutes. `go test -short` cuts it to a size that still exercises every
// generator branch, and the full number is what the report quotes.
const nPrograms = 500
const nProgramsShort = 40

func corpusSize() int {
	if testing.Short() {
		return nProgramsShort
	}
	return nPrograms
}

func TestDifferentialForward(t *testing.T) {
	requireBackend(t)
	requireCompiler(t)
	r := rand.New(rand.NewSource(7))
	var exactChecked, tolChecked, skipped int
	worstTol := 0.0
	n := corpusSize()
	for i := 0; i < n; i++ {
		p, err := generate(r, 6+r.Intn(10))
		if err != nil {
			skipped++
			continue
		}
		want, err := ir.Eval(p.g, p.args)
		if err != nil {
			skipped++
			continue
		}
		if !finite(want) {
			skipped++ // an overflowed program tests nothing about the compiler
			continue
		}
		var unfused []*tensor.Tensor
		for _, fuse := range []bool{false, true} {
			prog, err := codegen.Compile(p.g, codegen.Options{Fuse: fuse})
			if err != nil {
				t.Fatalf("program %d: compile(fuse=%v): %v\n%s", i, fuse, err, p.g)
			}
			got, err := prog.Run(p.args)
			if err != nil {
				prog.Close()
				t.Fatalf("program %d: run(fuse=%v): %v", i, fuse, err)
			}
			prog.Close()

			for k := range want {
				e, at := worstRel(got[k].Data, want[k].Data)
				bar := bitExact
				if p.inexact {
					bar = transTol
				}
				if e > bar {
					t.Fatalf("program %d, output %d, fuse=%v: element %d is %v, interpreter says %v (relative %g, bar %g)\n%s",
						i, k, fuse, at, got[k].Data[at], want[k].Data[at], e, bar, p.g)
				}
				if e > worstTol {
					worstTol = e
				}
			}
			// Fused against unfused is a second and stricter comparison: it is
			// bit-exact even for the transcendental programs, because both go
			// through the same libm.
			if !fuse {
				unfused = got
			} else {
				for k := range got {
					if e, at := worstRel(got[k].Data, unfused[k].Data); at >= 0 {
						t.Fatalf("program %d, output %d: fused and unfused disagree at element %d (relative %g)\n%s",
							i, k, at, e, p.g)
					}
				}
			}
		}
		if p.inexact {
			tolChecked++
		} else {
			exactChecked++
		}
	}
	t.Logf("%d programs generated: %d compared bit-exact, %d compared under %g, %d skipped (non-finite or unbuildable)",
		n, exactChecked, tolChecked, transTol, skipped)
	t.Logf("every program, transcendental or not, compared fused against unfused bit-exact")
	t.Logf("worst relative disagreement over the tolerance-bounded programs: %.3g", worstTol)
	if exactChecked == 0 || tolChecked == 0 {
		t.Fatalf("the generator did not cover both tolerance classes")
	}
}

// The gate for gradients. The compiled backward pass has to agree with
// tensor.Backward, which is what `grad` in a twill program actually runs.
func TestDifferentialGradient(t *testing.T) {
	requireBackend(t)
	requireCompiler(t)
	r := rand.New(rand.NewSource(11))
	var exactChecked, tolChecked, skipped int
	worst := 0.0
	n := corpusSize()
	for i := 0; i < n; i++ {
		p, err := generate(r, 6+r.Intn(8))
		if err != nil {
			skipped++
			continue
		}
		fwd, err := ir.Eval(p.g, p.args)
		if err != nil || !finite(fwd) {
			skipped++
			continue
		}
		// A fixed irregular cotangent per output. An all-ones cotangent makes a
		// whole class of index-shuffling bugs invisible, which is the reason
		// fullgradcheck_test.go seeds one this way too.
		ct := make([]*tensor.Tensor, len(fwd))
		for k, o := range fwd {
			d := make([]float64, len(o.Data))
			for j := range d {
				d[j] = math.Sin(float64(j)*1.7+0.3) * (1 + 0.25*float64(j%5))
			}
			ct[k] = tensor.New(d, o.Shape)
		}
		_, want, err := ir.EvalWithGrad(p.g, p.args, ct)
		if err != nil {
			skipped++
			continue
		}
		gg, err := ir.Grad(p.g)
		if err != nil {
			skipped++
			continue
		}
		args := append(append([]*tensor.Tensor{}, p.args...), ct...)
		var unfused []*tensor.Tensor
		for _, fuse := range []bool{false, true} {
			prog, err := codegen.Compile(gg, codegen.Options{Fuse: fuse})
			if err != nil {
				t.Fatalf("program %d: compile backward (fuse=%v): %v", i, fuse, err)
			}
			got, err := prog.Run(args)
			prog.Close()
			if err != nil {
				t.Fatalf("program %d: run backward (fuse=%v): %v", i, fuse, err)
			}
			if !fuse {
				unfused = got
				continue
			}
			// Fused against unfused: no exceptions and no tolerance. Both go
			// through the same libm and accumulate in the same order, so a single
			// differing bit is a fusion defect.
			for k := range p.args {
				a, b := got[len(p.g.Out)+k], unfused[len(p.g.Out)+k]
				if e, at := worstRel(a.Data, b.Data); at >= 0 {
					t.Fatalf("program %d, d/d%s: fused and unfused gradients disagree at element %d (relative %g)\n%s",
						i, p.g.Params[k].Name, at, e, gg)
				}
			}
			for k := range p.args {
				g := got[len(p.g.Out)+k]
				if !finite([]*tensor.Tensor{g}) {
					continue
				}
				e, at := worstRel(g.Data, want[k])
				if e > gradTol {
					t.Fatalf("program %d, d/d%s: element %d is %v, interpreter says %v (relative %g, bar %g)\n%s",
						i, p.g.Params[k].Name, at, g.Data[at], want[k][at], e, gradTol, gg)
				}
				if at < 0 {
					exactChecked++
				} else {
					tolChecked++
				}
				if e > worst {
					worst = e
				}
			}
		}
	}
	t.Logf("%d programs differentiated, %d skipped (non-finite or no VJP rule)", n, skipped)
	t.Logf("every fused gradient matched its unfused counterpart bit for bit")
	t.Logf("against tensor.Backward: %d parameter gradients bit-identical, %d differing within %g",
		exactChecked, tolChecked, gradTol)
	t.Logf("worst relative disagreement on a gradient: %.3g", worst)
	if exactChecked == 0 {
		t.Fatalf("nothing came out bit-identical, which is not what the accumulation-order argument predicts")
	}
}

// TestTranscendentalULP is the measurement the tolerance is sited on. It runs
// each function twice over its useful range, once through Go and once through
// the emitted code, and reports the worst disagreement in units in the last
// place. It is not an assertion about correctness; it is the evidence for the
// number transTol is set to, and it fails only if the machine's libm is far
// enough off that the bound would be fitted to it rather than chosen above it.
func TestTranscendentalULP(t *testing.T) {
	requireBackend(t)
	requireCompiler(t)
	type fn struct {
		name string
		op   ir.Op
		lo   float64
		hi   float64
		gof  func(float64) float64
	}
	fns := []fn{
		{"exp", ir.OpExp, -20, 20, math.Exp},
		{"log", ir.OpLog, 1e-6, 1e6, math.Log},
		{"sin", ir.OpSin, -30, 30, math.Sin},
		{"cos", ir.OpCos, -30, 30, math.Cos},
		{"tanh", ir.OpTanh, -12, 12, math.Tanh},
		{"sqrt", ir.OpSqrt, 0, 1e6, math.Sqrt},
	}
	const n = 20001
	for _, f := range fns {
		b := ir.NewBuilder()
		x := b.Param("x", []int{n})
		b.Output(b.Unary(f.op, x))
		g, err := b.Finish()
		if err != nil {
			t.Fatal(err)
		}
		data := make([]float64, n)
		for i := range data {
			data[i] = f.lo + (f.hi-f.lo)*float64(i)/float64(n-1)
		}
		prog := compile(t, g, true)
		got, err := prog.Run([]*tensor.Tensor{tensor.New(data, []int{n})})
		if err != nil {
			t.Fatal(err)
		}
		worstUlp, worstRel, differing := 0.0, 0.0, 0
		for i, v := range data {
			w := f.gof(v)
			c := got[0].Data[i]
			if math.Float64bits(w) == math.Float64bits(c) {
				continue
			}
			differing++
			u := math.Abs(float64(int64(math.Float64bits(c)) - int64(math.Float64bits(w))))
			if u > worstUlp {
				worstUlp = u
			}
			if e := relErr(c, w); e > worstRel {
				worstRel = e
			}
		}
		t.Logf("%-5s over [%g, %g], %d points: %d differ from Go, worst %.0f ulp, worst relative %.3g",
			f.name, f.lo, f.hi, n, differing, worstUlp, worstRel)
		if worstRel > ulpRel/2 {
			t.Errorf("%s disagrees by %g, which is more than half the per-primitive bound; "+
				"ulpRel is meant to sit above the measurement with room, so either raise it "+
				"deliberately or find out what this toolchain's libm is doing", f.name, worstRel)
		}
	}
}
