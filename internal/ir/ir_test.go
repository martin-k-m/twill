package ir_test

import (
	"math"
	"testing"

	"github.com/martin-k-m/twill/internal/ir"
	"github.com/martin-k-m/twill/internal/tensor"
)

// The IR's first claim is that it does not change any arithmetic: evaluating a
// graph node by node has to give exactly what calling the same internal/tensor
// functions in the same order gives, to the last bit. That is the claim these
// tests make, before any kernel exists to be blamed for a difference.

func seq(n int, f func(i int) float64) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = f(i)
	}
	return d
}

func bitEqual(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("%s: element %d is %v, want %v (bits %#x vs %#x)",
				name, i, got[i], want[i], math.Float64bits(got[i]), math.Float64bits(want[i]))
		}
	}
}

// mcGraph is the Monte Carlo European call from examples/montecarlo_option.tw,
// as IR. It is the program docs/CODEGEN.md picks as the primary benchmark, and
// it is the shape fusion exists for: one elementwise chain into a reduction.
// Params are (S0, sigma) so the graph is differentiable in exactly the two
// inputs the example takes delta and vega with respect to.
func mcGraph(paths int, z []float64) (*ir.Graph, error) {
	b := ir.NewBuilder()
	S0 := b.Param("S0", []int{})
	sigma := b.Param("sigma", []int{})
	Z := b.Const(z, []int{paths})
	r := b.Scalar(0.05)
	T := b.Scalar(1.0)
	K := b.Scalar(100.0)
	half := b.Scalar(0.5)

	// drift = (r - 0.5*sigma*sigma) * T
	s2 := b.Binary(ir.OpMul, b.Binary(ir.OpMul, half, sigma), sigma)
	drift := b.Binary(ir.OpMul, b.Binary(ir.OpSub, r, s2), T)
	// ST = S0 * exp(drift + sigma*sqrt(T)*Z)
	vol := b.Binary(ir.OpMul, b.Binary(ir.OpMul, sigma, b.Unary(ir.OpSqrt, T)), Z)
	ST := b.Binary(ir.OpMul, S0, b.Unary(ir.OpExp, b.Binary(ir.OpAdd, drift, vol)))
	payoff := b.Unary(ir.OpRelu, b.Binary(ir.OpSub, ST, K))
	disc := b.Unary(ir.OpExp, b.Unary(ir.OpNeg, b.Binary(ir.OpMul, r, T)))
	price := b.Binary(ir.OpMul, disc, b.Mean(payoff))
	b.Output(price)
	return b.Finish()
}

func mcDirect(paths int, z []float64, s0v, sigv float64) *tensor.Tensor {
	S0 := tensor.Scalar(s0v)
	sigma := tensor.Scalar(sigv)
	Z := tensor.New(append([]float64(nil), z...), []int{paths})
	r := tensor.Scalar(0.05)
	T := tensor.Scalar(1.0)
	K := tensor.Scalar(100.0)
	half := tensor.Scalar(0.5)
	must := func(t *tensor.Tensor, err error) *tensor.Tensor {
		if err != nil {
			panic(err)
		}
		return t
	}
	s2 := must(tensor.Mul(must(tensor.Mul(half, sigma)), sigma))
	drift := must(tensor.Mul(must(tensor.Sub(r, s2)), T))
	vol := must(tensor.Mul(must(tensor.Mul(sigma, tensor.Sqrt(T))), Z))
	ST := must(tensor.Mul(S0, tensor.Exp(must(tensor.Add(drift, vol)))))
	payoff := tensor.Relu(must(tensor.Sub(ST, K)))
	disc := tensor.Exp(tensor.Neg(must(tensor.Mul(r, T))))
	return must(tensor.Mul(disc, tensor.Mean(payoff)))
}

func TestEvalIsBitIdenticalToTensor(t *testing.T) {
	const paths = 20000
	z := seq(paths, func(i int) float64 { return math.Sin(float64(i)*0.7919 + 0.11) })
	g, err := mcGraph(paths, z)
	if err != nil {
		t.Fatal(err)
	}
	out, err := ir.Eval(g, []*tensor.Tensor{tensor.Scalar(100), tensor.Scalar(0.2)})
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, "mc price", out[0].Data, mcDirect(paths, z, 100, 0.2).Data)
}

// The Black-Scholes closed form is the check that does not go through the
// interpreter at all, so it catches a shared mistake the differential tests
// could not. It is a tolerance check because Monte Carlo converges at 1/sqrt(n)
// and 200,000 paths of a deterministic sinusoidal stand-in for randn is not a
// sample from a normal distribution at all; the assertion is only that the
// pricer is in the right neighbourhood, not that it is accurate.
func TestMonteCarloIsPlausible(t *testing.T) {
	const paths = 200000
	// A crude inverse-CDF sample so the shocks really are standard normal.
	z := seq(paths, func(i int) float64 {
		u := (float64(i) + 0.5) / float64(paths)
		return math.Sqrt2 * erfinv(2*u-1)
	})
	g, err := mcGraph(paths, z)
	if err != nil {
		t.Fatal(err)
	}
	out, err := ir.Eval(g, []*tensor.Tensor{tensor.Scalar(100), tensor.Scalar(0.2)})
	if err != nil {
		t.Fatal(err)
	}
	price := out[0].Data[0]
	const bs = 10.4506
	if math.Abs(price-bs) > 0.05 {
		t.Fatalf("price %v is not near the Black-Scholes value %v", price, bs)
	}
}

// erfinv is a small rational approximation, good to about 4e-4 relative, which
// is far tighter than the 0.05 tolerance above needs.
func erfinv(x float64) float64 {
	const a = 0.147
	ln := math.Log(1 - x*x)
	t1 := 2/(math.Pi*a) + ln/2
	return math.Copysign(math.Sqrt(math.Sqrt(t1*t1-ln/a)-t1), x)
}

func TestFusionOnTheMonteCarloChain(t *testing.T) {
	const paths = 200000
	z := seq(paths, func(i int) float64 { return math.Sin(float64(i)) })
	g, err := mcGraph(paths, z)
	if err != nil {
		t.Fatal(err)
	}
	off := ir.FuseOff(g).Stats()
	on := ir.Fuse(g).Stats()
	t.Logf("forward, fusion off: %s", off)
	t.Logf("forward, fusion on:  %s", on)
	if on.Kernels >= off.Kernels {
		t.Fatalf("greedy fusion did not reduce the kernel count: %d vs %d", on.Kernels, off.Kernels)
	}
	// The whole point: the 200,000-element intermediates stop existing.
	if on.Bytes >= off.Bytes/2 {
		t.Fatalf("fusion left %d B of intermediates against %d B unfused", on.Bytes, off.Bytes)
	}

	gg, err := ir.Grad(g)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("fwd+bwd, fusion off: %s", ir.FuseOff(gg).Stats())
	t.Logf("fwd+bwd, fusion on:  %s", ir.Fuse(gg).Stats())
}

// The gradient transform's claim: differentiating the IR gives what
// tensor.Backward gives on the same computation. This is the sharpest oracle
// available for it, because it is exact rather than a finite difference.
func TestGradTransformMatchesTensorBackward(t *testing.T) {
	const paths = 4096
	z := seq(paths, func(i int) float64 { return math.Sin(float64(i)*0.31 + 0.2) })
	g, err := mcGraph(paths, z)
	if err != nil {
		t.Fatal(err)
	}
	args := []*tensor.Tensor{tensor.Scalar(100), tensor.Scalar(0.2)}
	ct := []*tensor.Tensor{tensor.Scalar(1)}

	_, want, err := ir.EvalWithGrad(g, args, ct)
	if err != nil {
		t.Fatal(err)
	}

	gg, err := ir.Grad(g)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ir.Eval(gg, append(append([]*tensor.Tensor{}, args...), ct...))
	if err != nil {
		t.Fatal(err)
	}
	// Outputs of the transformed graph are (price, dS0, dsigma).
	if len(got) != 1+len(args) {
		t.Fatalf("transformed graph returned %d outputs, want %d", len(got), 1+len(args))
	}
	for i := range args {
		bitEqual(t, g.Params[i].Name, got[1+i].Data, want[i])
	}
	t.Logf("delta = %v, vega = %v", got[1].Data[0], got[2].Data[0])
}
