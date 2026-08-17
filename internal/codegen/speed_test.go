package codegen_test

import (
	"math"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/twill-lang/twill/internal/ir"
	"github.com/twill-lang/twill/internal/tensor"
)

// A speed comparison, run only when asked for, and interleaved.
//
// docs/BENCHMARKS.md section 7 measured 74% thermal drift on this class of
// machine, biasing back-to-back comparisons by 15 to 40%. So the two
// implementations alternate within one loop rather than being measured in
// blocks, which is the correction bench/cmd/twillbench already adopted. Median
// and p99 over 30 timed iterations after 5 warmups, matching that harness.
//
// This is not the benchmark docs/CODEGEN.md section 8 specifies. That one runs
// bench/workloads/mc_option_grad.tw through the real CLI, and it cannot run
// here because there is no front end that turns a .tw file into IR. What is
// measured here is the same mathematics driven from Go: the interpreter path is
// ir.Eval, which calls exactly the internal/tensor functions the interpreter
// calls, against the compiled path.
func TestSpeedAgainstInterpreter(t *testing.T) {
	if os.Getenv("TWILL_SPEED") == "" {
		t.Skip("set TWILL_SPEED=1 to run the speed comparison")
	}
	requireCompiler(t)
	const paths = 200000
	z := make([]float64, paths)
	for i := range z {
		z[i] = math.Sin(float64(i)*0.7919 + 0.11)
	}
	fwd := mcBenchGraph(t, paths, z)
	bwd, err := ir.Grad(fwd)
	if err != nil {
		t.Fatal(err)
	}
	args := []*tensor.Tensor{tensor.Scalar(100), tensor.Scalar(0.2)}
	ct := []*tensor.Tensor{tensor.Scalar(1)}

	for _, c := range []struct {
		name string
		g    *ir.Graph
		args []*tensor.Tensor
	}{
		{"mc_option_fwd", fwd, args},
		{"mc_option_grad", bwd, append(append([]*tensor.Tensor{}, args...), ct...)},
	} {
		prog := compile(t, c.g, true)
		var interp, comp []float64
		for i := 0; i < 35; i++ {
			t0 := time.Now()
			if _, err := ir.Eval(c.g, c.args); err != nil {
				t.Fatal(err)
			}
			d0 := time.Since(t0)
			t1 := time.Now()
			if _, err := prog.Run(c.args); err != nil {
				t.Fatal(err)
			}
			d1 := time.Since(t1)
			if i >= 5 {
				interp = append(interp, d0.Seconds()*1000)
				comp = append(comp, d1.Seconds()*1000)
			}
		}
		sort.Float64s(interp)
		sort.Float64s(comp)
		med := func(v []float64) float64 { return v[len(v)/2] }
		p99 := func(v []float64) float64 { return v[int(float64(len(v))*0.99)-1] }
		t.Logf("%s: ir.Eval  median %.3f ms  p99 %.3f ms", c.name, med(interp), p99(interp))
		t.Logf("%s: compiled  median %.3f ms  p99 %.3f ms  (%.2fx)",
			c.name, med(comp), p99(comp), med(interp)/med(comp))
		t.Logf("%s: arena %d f64, %s", c.name, prog.ArenaSize(), prog.Plan().Stats())
	}
}

func mcBenchGraph(t *testing.T, paths int, z []float64) *ir.Graph {
	t.Helper()
	b := ir.NewBuilder()
	S0 := b.Param("S0", []int{})
	sigma := b.Param("sigma", []int{})
	Z := b.Const(z, []int{paths})
	r := b.Scalar(0.05)
	T := b.Scalar(1.0)
	K := b.Scalar(100.0)
	half := b.Scalar(0.5)
	s2 := b.Binary(ir.OpMul, b.Binary(ir.OpMul, half, sigma), sigma)
	drift := b.Binary(ir.OpMul, b.Binary(ir.OpSub, r, s2), T)
	vol := b.Binary(ir.OpMul, b.Binary(ir.OpMul, sigma, b.Unary(ir.OpSqrt, T)), Z)
	ST := b.Binary(ir.OpMul, S0, b.Unary(ir.OpExp, b.Binary(ir.OpAdd, drift, vol)))
	payoff := b.Unary(ir.OpRelu, b.Binary(ir.OpSub, ST, K))
	disc := b.Unary(ir.OpExp, b.Unary(ir.OpNeg, b.Binary(ir.OpMul, r, T)))
	b.Output(b.Binary(ir.OpMul, disc, b.Mean(payoff)))
	g, err := b.Finish()
	if err != nil {
		t.Fatal(err)
	}
	return g
}
