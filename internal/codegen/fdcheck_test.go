package codegen_test

import (
	"math"
	"sort"
	"testing"

	"github.com/martin-k-m/twill/internal/codegen"
	"github.com/martin-k-m/twill/internal/ir"
	"github.com/martin-k-m/twill/internal/tensor"
)

// Gradient checking the *compiled* backward pass against finite differences.
//
// The differential tests compare the compiler against the interpreter. If the
// interpreter is wrong they agree and both are wrong, which docs/CODEGEN.md
// section 7.4 names as the gap. This file is the independent check that closes
// it for gradients: nothing here consults the interpreter's autodiff at all.
// The forward values come from the compiled program and the derivative comes
// from the compiled backward pass, and both are compared against a numerical
// derivative of the compiled forward pass.
//
// The method is the one in internal/tensor/fullgradcheck_test.go, reused rather
// than reinvented, down to the constants:
//
//	D(h) = (L(x+h) - L(x-h)) / 2h              truncation O(h^2)
//	D*   = (4*D(h/2) - D(h)) / 3               truncation O(h^4)
//
// with h = 1e-4 * max(1, |x_i|) and a 1e-7 relative bar. The extrapolation is
// what makes that bar honest: a plain central difference carries about 1e-10 of
// its own error, close enough to a real small gradient bug to make a failure
// ambiguous, and D* lands near 1e-12.
//
// The cotangent is the same irregular one fullgradcheck uses, for the same
// reason: an all-ones cotangent makes a permutation of the output invisible.
//
// Kinks. relu, clip, maximum, minimum and where are piecewise, and at a kink a
// central difference straddles two branches and reports the average of the
// one-sided slopes, which is not what any autodiff system returns. Every case
// below is sited away from its kinks, exactly as the cases in fullgradcheck are.

const (
	fdHRel = 1e-4
	fdTol  = 1e-7
	// absFloor keeps a true derivative of zero from failing on relative error
	// against a numeric 1e-13.
	fdAbsFloor = 1e-9
)

type fdCase struct {
	name  string
	data  []float64
	shape []int
	build func(b *ir.Builder, x ir.Ref) ir.Ref
	tol   float64
}

func fdCases() []fdCase {
	v4 := []float64{0.7, -1.3, 2.1, -0.4}
	pos4 := []float64{0.4, 1.7, 2.9, 0.9}
	m23 := []float64{0.5, -1.2, 0.9, 2.3, -0.7, 1.6}
	m32 := []float64{0.5, -1.2, 0.9, 2.3, -0.7, 1.6}

	other4 := func(b *ir.Builder) ir.Ref {
		return b.Const([]float64{1.1, -0.6, 0.3, 2.7}, []int{4})
	}
	row3 := func(b *ir.Builder) ir.Ref { return b.Const([]float64{2.0, -1.0, 0.5}, []int{3}) }
	col2 := func(b *ir.Builder) ir.Ref { return b.Const([]float64{1.5, -2.5}, []int{2, 1}) }

	bin := func(name string, op ir.Op) fdCase {
		return fdCase{name: name, data: v4, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Binary(op, x, other4(b)) }}
	}
	un := func(name string, op ir.Op, data []float64) fdCase {
		return fdCase{name: name, data: data, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Unary(op, x) }}
	}

	return []fdCase{
		bin("add", ir.OpAdd), bin("sub", ir.OpSub), bin("mul", ir.OpMul),
		// maximum and minimum against a constant with no ties anywhere.
		bin("maximum", ir.OpMaximum), bin("minimum", ir.OpMinimum),
		{name: "div", data: v4, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Binary(ir.OpDiv, x, other4(b)) }},
		{name: "div/x-in-divisor", data: pos4, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Binary(ir.OpDiv, other4(b), x) }},

		un("neg", ir.OpNeg, v4), un("square", ir.OpSquare, v4),
		un("exp", ir.OpExp, v4), un("log", ir.OpLog, pos4), un("sqrt", ir.OpSqrt, pos4),
		un("sin", ir.OpSin, v4), un("cos", ir.OpCos, v4),
		un("tanh", ir.OpTanh, v4), un("sigmoid", ir.OpSigmoid, v4),
		// relu away from zero on every coordinate.
		un("relu", ir.OpRelu, []float64{0.7, -1.3, 2.1, -0.4}),
		{name: "pow_scalar", data: pos4, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.PowScalar(x, 2.5) }},
		// clip with no coordinate on a bound.
		{name: "clip", data: []float64{0.7, -1.3, 2.1, -0.4}, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Clip(x, -1.0, 1.5) }},
		{name: "where", data: v4, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref {
				c := b.Const([]float64{1, 0, 1, 0}, []int{4})
				return b.Where(c, b.Unary(ir.OpSquare, x), b.Binary(ir.OpMul, x, b.Scalar(3)))
			}},

		// broadcasting regimes, which is where a wrong sum_to shows up.
		{name: "add/scalar", data: v4, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Binary(ir.OpAdd, x, b.Scalar(2.5)) }},
		{name: "mul/row-broadcast", data: m23, shape: []int{2, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Binary(ir.OpMul, x, row3(b)) }},
		{name: "mul/col-broadcast", data: m23, shape: []int{2, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Binary(ir.OpMul, x, col2(b)) }},
		{name: "mul/scalar-x", data: []float64{1.4}, shape: []int{},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Binary(ir.OpMul, x, other4(b)) }},

		// reductions.
		{name: "sum", data: v4, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Sum(x) }},
		{name: "mean", data: v4, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Mean(x) }},
		{name: "sum_axis/0", data: m23, shape: []int{2, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.SumAxis(x, 0) }},
		{name: "sum_axis/1", data: m23, shape: []int{2, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.SumAxis(x, 1) }},
		{name: "mean_axis/1", data: m23, shape: []int{2, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.MeanAxis(x, 1) }},
		{name: "sum_to", data: m23, shape: []int{2, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.SumTo(x, []int{3}) }},

		// structural.
		{name: "reshape", data: m23, shape: []int{2, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Reshape(x, []int{3, 2}) }},
		{name: "transpose", data: m23, shape: []int{2, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.Transpose(x, []int{1, 0}) }},
		{name: "broadcast_to", data: []float64{1.4, -0.6, 2.2}, shape: []int{1, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref { return b.BroadcastTo(x, []int{4, 3}) }},

		// contraction, on both sides.
		{name: "matmul/lhs", data: m23, shape: []int{2, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref {
				return b.MatMul(x, b.Const(m32, []int{3, 2}))
			}},
		{name: "matmul/rhs", data: m32, shape: []int{3, 2},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref {
				return b.MatMul(b.Const(m23, []int{2, 3}), x)
			}},
		{name: "matmul/vector", data: []float64{0.7, -1.3, 2.1}, shape: []int{3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref {
				return b.MatMul(b.Const(m23, []int{2, 3}), x)
			}},

		// a chain, so a wrong saved intermediate has somewhere to hide.
		{name: "chain/mc-payoff", data: []float64{0.7, -1.3, 2.1, -0.4}, shape: []int{4},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref {
				st := b.Binary(ir.OpMul, b.Scalar(100), b.Unary(ir.OpExp, b.Binary(ir.OpMul, x, b.Scalar(0.2))))
				return b.Mean(b.Unary(ir.OpRelu, b.Binary(ir.OpSub, st, b.Scalar(95))))
			}},
		{name: "chain/mlp-ish", data: m23, shape: []int{2, 3},
			build: func(b *ir.Builder, x ir.Ref) ir.Ref {
				h := b.Unary(ir.OpTanh, b.MatMul(x, b.Const(m32, []int{3, 2})))
				return b.Sum(b.Unary(ir.OpSquare, b.Binary(ir.OpSub, h, col2(b))))
			}},
	}
}

func cotangent(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = math.Sin(float64(i)*1.7+0.3) * (1 + 0.25*float64(i%5))
	}
	return w
}

func TestCompiledGradientAgainstFiniteDifferences(t *testing.T) {
	requireCompiler(t)
	cases := fdCases()
	type result struct {
		name string
		err  float64
	}
	results := make([]result, 0, len(cases))

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			b := ir.NewBuilder()
			x := b.Param("x", c.shape)
			b.Output(c.build(b, x))
			g, err := b.Finish()
			if err != nil {
				t.Fatal(err)
			}
			fwd := compile(t, g, true)
			outShape := g.Nodes[g.Out[0]].Shape
			w := cotangent(ir.Numel(outShape))

			// L(x) = sum_i w_i * out_i, evaluated entirely by the compiled code.
			loss := func(data []float64) float64 {
				out, err := fwd.Run([]*tensor.Tensor{tensor.New(data, c.shape)})
				if err != nil {
					t.Fatal(err)
				}
				s := 0.0
				for i, v := range out[0].Data {
					s += w[i] * v
				}
				return s
			}

			gg, err := ir.Grad(g)
			if err != nil {
				t.Fatal(err)
			}
			bwd, err := codegen.Compile(gg, codegen.Options{Fuse: true})
			if err != nil {
				t.Fatal(err)
			}
			defer bwd.Close()
			got, err := bwd.Run([]*tensor.Tensor{
				tensor.New(append([]float64(nil), c.data...), c.shape),
				tensor.New(w, outShape),
			})
			if err != nil {
				t.Fatal(err)
			}
			analytic := got[1].Data

			tol := c.tol
			if tol == 0 {
				tol = fdTol
			}
			worst := 0.0
			for i := range c.data {
				h := fdHRel * math.Max(1, math.Abs(c.data[i]))
				d := func(step float64) float64 {
					up := append([]float64(nil), c.data...)
					dn := append([]float64(nil), c.data...)
					up[i] += step
					dn[i] -= step
					return (loss(up) - loss(dn)) / (2 * step)
				}
				numeric := (4*d(h/2) - d(h)) / 3
				e := math.Abs(analytic[i]-numeric) /
					math.Max(fdAbsFloor, math.Max(math.Abs(analytic[i]), math.Abs(numeric)))
				if e > worst {
					worst = e
				}
				if e > tol {
					t.Fatalf("coordinate %d: compiled gradient %v, finite difference %v (relative %g > %g)",
						i, analytic[i], numeric, e, tol)
				}
			}
			results = append(results, result{c.name, worst})
		})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].err > results[j].err })
	t.Logf("compiled gradient against Richardson-extrapolated central differences, %d cases, worst first:", len(results))
	for _, r := range results {
		t.Logf("  %-24s %.3e", r.name, r.err)
	}
}
