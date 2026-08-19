package interp_test

import (
	"math"
	"strings"
	"testing"

	"github.com/twill-lang/twill/internal/value"
)

// jvp, vjp and hvp are the primitives grad, jacobian and hessian are made of, so
// the checks here are all cross-checks: against a central difference, which
// shares no code with any of them, and against the five operations that already
// existed, which they have to agree with wherever the definitions overlap.
//
// Every test returns a single scalar -- the largest discrepancy it found -- and
// the Go side asserts it is small. That keeps the tolerance in one place and
// keeps a failure reporting how wrong it was rather than only that it was.

// fdPrelude is central differences over a parameter tree, written without grad
// so that agreeing with it is evidence rather than a tautology. h is 1e-5, whose
// truncation error is O(h^2) = 1e-10 and whose rounding error is O(eps/h) = 1e-11
// on f64, so 1e-6 is a tolerance the method can actually meet.
const fdPrelude = `
let H = 0.00001
fn bump(x, v, s) = zip_leaves(fn(p) = p[0] + p[1] * s, list(x, v))
fn nudge(x, v, s) = bump(x, v, s * H)
# The directional derivative of a scalar f along v, by central difference.
fn fd(f, x, v) = (f(nudge(x, v, 1.0)) - f(nudge(x, v, -1.0))) / (2.0 * H)
# The same for a tensor-valued f, elementwise.
fn fdt(f, x, v) = (f(nudge(x, v, 1.0)) - f(nudge(x, v, -1.0))) / (2.0 * H)
# The second directional derivative vᵀHv, by a second central difference. Its
# step is 1e-3 rather than 1e-5 because the quotient divides by h², where a
# smaller step loses to cancellation faster than it gains in truncation.
fn fd2(f, x, v) = (f(bump(x, v, 0.001)) - 2.0 * f(x) + f(bump(x, v, -0.001))) / 0.000001
fn worst(a, b) = max(abs(a - b) / maximum(maximum(abs(a), abs(b)), 1.0))
`

func maxErr(t *testing.T, src string) float64 {
	t.Helper()
	v, _ := run(t, fdPrelude+src)
	tn, ok := value.AsTensor(v)
	if !ok || len(tn.Data) != 1 {
		t.Fatalf("expected a scalar discrepancy, got %v", v)
	}
	return tn.Data[0]
}

func expectSmall(t *testing.T, name, src string, tol float64) {
	t.Helper()
	got := maxErr(t, src)
	if math.IsNaN(got) || got > tol {
		t.Errorf("%s: worst discrepancy %g, want <= %g", name, got, tol)
	}
}

// --- jvp against finite differences ----------------------------------------

func TestJVPAgainstFiniteDifferencesScalarOutput(t *testing.T) {
	cases := []struct{ name, body, x, v string }{
		{"scalar-input", "fn f(x) = sin(x) * exp(x * 0.5)", "1.3", "0.7"},
		{"vector", "fn f(x) = sum(square(x) * [1.0, 2.0, 3.0]) + sum(tanh(x))", "[0.4, -1.1, 2.0]", "[0.5, -1.0, 2.0]"},
		{"matrix", "fn f(A) = sum(log(square(A @ A) + 2.0))", "[[1.0, 2.0], [0.3, -0.7]]", "[[0.2, -0.4], [1.0, 0.1]]"},
		{"softmax", "fn f(x) = sum(square(softmax(x, 0)))", "[0.5, -1.0, 2.0, 0.3]", "[1.0, 0.25, -0.5, 0.75]"},
		{"reduction-chain", "fn f(x) = logsumexp(x, 0) * mean(sqrt(square(x) + 1.0))", "[0.5, -1.0, 2.0, 0.3]", "[1.0, 0.25, -0.5, 0.75]"},
	}
	for _, c := range cases {
		src := c.body + "\nlet x = " + c.x + "\nlet v = " + c.v + `
let got = jvp(f)(x, v)
worst(got[0], f(x)) + worst(got[1], fd(f, x, v))`
		expectSmall(t, "jvp/"+c.name, src, 1e-6)
	}
}

// A record of parameters is the shape a model has, and grad follows it, so jvp
// has to take a tangent with the same fields and follow it too.
func TestJVPOverAParameterRecord(t *testing.T) {
	src := `
let X = [[1.0, 2.0], [0.5, 1.5], [-1.0, 0.25]]
fn f(p) = sum(square(tanh(X @ p.w + p.b))) + sum(p.b)
let p = { w: [[0.3, -0.6], [1.1, 0.2]], b: [0.5, -0.25] }
let v = { w: [[1.0, 0.5], [-0.25, 2.0]], b: [0.75, -1.5] }
let got = jvp(f)(p, v)
worst(got[0], f(p)) + worst(got[1], fd(f, p, v))`
	expectSmall(t, "jvp/record", src, 1e-6)
}

// A list of tensors is the other parameter tree, and the pieces do not have to
// be the same shape.
func TestJVPOverANestedList(t *testing.T) {
	src := `
let x = list([1.0, 2.0], [0.3, -0.7, 1.5])
fn f(x) = sum(square(x[0])) + sum(exp(x[1] * 0.5))
let v = list([1.0, -0.5], [0.25, 2.0, -1.0])
let got = jvp(f)(x, v)
worst(got[0], f(x)) + worst(got[1], fd(f, x, v))`
	expectSmall(t, "jvp/list", src, 1e-6)
}

// A tensor-valued f: the tangent has the shape of the output, and it is the
// Jacobian times v, so it must equal jacobian(f)(x) @ v.
func TestJVPTensorOutputMatchesJacobianTimesV(t *testing.T) {
	src := `
fn f(x) {
  let a = x[0:1] * x[1:2]
  let b = sin(x[1:2] * x[2:3])
  let c = exp(x[0:1] * 0.5)
  concat([a, b, c], 0)
}
let x = [0.7, -1.2, 2.1]
let v = [0.4, 1.5, -0.9]
let got = jvp(f)(x, v)
worst(got[0], f(x)) + worst(got[1], jacobian(f)(x) @ v) + worst(got[1], fdt(f, x, v))`
	expectSmall(t, "jvp/tensor-output", src, 1e-6)
}

// --- vjp against grad, jacobian and finite differences ----------------------

// grad(f)(x) is vjp with a cotangent of 1 on a scalar output. That is the
// definition, so it is the first thing to check, and it checks vjp's tree
// handling at the same time because grad already follows records.
func TestVJPWithUnitCotangentIsGrad(t *testing.T) {
	src := `
let X = [[1.0, 2.0], [0.5, 1.5], [-1.0, 0.25]]
fn f(p) = sum(square(tanh(X @ p.w + p.b))) + sum(exp(p.b * 0.5))
let p = { w: [[0.3, -0.6], [1.1, 0.2]], b: [0.5, -0.25] }
let got = vjp(f)(p, 1.0)
let g = grads(f)(p)[0]
worst(got[0], f(p)) + worst(got[1].w, g.w) + worst(got[1].b, g.b)`
	expectSmall(t, "vjp/is-grad", src, 1e-12)
}

// For a tensor-valued f the cotangent picks a row combination of the Jacobian,
// so vjp(f)(x, v) must equal vᵀ J. Checked against jacobian, and against a
// finite difference of the contracted scalar, which shares no code with either.
func TestVJPMatchesJacobianTransposeTimesV(t *testing.T) {
	src := `
fn f(x) {
  let a = x[0:1] * x[1:2]
  let b = sin(x[1:2] * x[2:3])
  let c = exp(x[0:1] * 0.5)
  concat([a, b, c], 0)
}
let x = [0.7, -1.2, 2.1]
let v = [1.3, -0.5, 2.0]
let got = vjp(f)(x, v)
let J = jacobian(f)(x)
fn contracted(y) = sum(f(y) * v)
worst(got[0], f(x)) + worst(got[1], transpose(J) @ v) + worst(got[1], grad(contracted)(x))`
	expectSmall(t, "vjp/jacobian-transpose", src, 1e-12)
}

// jvp and vjp are adjoint: vᵀ (J u) = (Jᵀ v)ᵀ u for any u and v. Neither is
// computed from the other, so agreement here is a check of both at once, and it
// is the identity that would break first if either had a transposed index.
func TestJVPAndVJPAreAdjoint(t *testing.T) {
	src := `
fn f(x) {
  let a = x[0:1] * x[1:2] * x[2:3]
  let b = tanh(x[1:2]) + x[0:1]
  concat([a, b], 0)
}
let x = [0.7, -1.2, 2.1]
let u = [0.4, 1.5, -0.9]
let v = [1.3, -0.5]
let forward = sum(jvp(f)(x, u)[1] * v)
let reverse = sum(vjp(f)(x, v)[1] * u)
abs(forward - reverse)`
	expectSmall(t, "jvp/vjp-adjoint", src, 1e-12)
}

// A dense layer and a two-layer MLP: the shape the operations actually get used
// in, with a record of parameters and a batch of inputs.
func TestVJPAndJVPOnAnMLPForward(t *testing.T) {
	src := `
let X = [[0.4, -1.1, 0.7], [1.2, 0.3, -0.5]]
let y = [1.0, 0.0]
fn mlp(p) {
  let h = tanh(X @ p.w1 + p.b1)
  let o = sigmoid(h @ p.w2 + p.b2)
  mean(square(reshape(o, list(2)) - y))
}
let p = {
  w1: [[0.5, -0.2, 0.9, 0.1], [0.3, 0.7, -0.4, 0.2], [-0.6, 0.15, 0.25, -0.8]],
  b1: [0.05, -0.1, 0.2, 0.0],
  w2: [[0.4], [-0.7], [0.25], [0.9]],
  b2: [0.1],
}
let v = map_leaves(fn(t) = sin(t * 7.0 + 1.0), p)
let g = grads(mlp)(p)[0]
let fw = jvp(mlp)(p, v)
let rv = vjp(mlp)(p, 1.0)
# The forward tangent is the gradient contracted with v, the reverse cotangent
# is the gradient itself, and the finite difference agrees with the tangent.
let contracted = sum(g.w1 * v.w1) + sum(g.b1 * v.b1) + sum(g.w2 * v.w2) + sum(g.b2 * v.b2)
worst(fw[0], mlp(p)) + worst(fw[1], contracted) + worst(fw[1], fd(mlp, p, v))
  + worst(rv[1].w1, g.w1) + worst(rv[1].b1, g.b1)
  + worst(rv[1].w2, g.w2) + worst(rv[1].b2, g.b2)`
	expectSmall(t, "mlp", src, 1e-6)
}

// --- hvp --------------------------------------------------------------------

// The Hessian of the quadratic form xᵀAx is A + Aᵀ at every point, so H v is a
// number that can be written down without differentiating anything.
func TestHVPAgainstAHandComputedHessian(t *testing.T) {
	got := maxErr(t, `
let A = [[2.0, 0.5, -1.0], [0.5, 6.0, 0.25], [-1.0, 0.25, 3.0]]
fn f(x) = sum((A @ x) * x)
let x = [1.0, -2.0, 0.5]
let v = [0.4, 1.5, -0.9]
let want = (A + transpose(A)) @ v
worst(hvp(f)(x, v), want)`)
	if got > 1e-14 {
		t.Errorf("hvp on a quadratic form: discrepancy %g, want <= 1e-14", got)
	}
}

// hvp must be hessian(f)(x) @ v -- the same number by a cheaper route -- and the
// directional second derivative vᵀHv must match a central difference of the
// gradient along v.
func TestHVPMatchesHessianTimesV(t *testing.T) {
	cases := []struct{ name, body, x, v string }{
		{"quartic", "fn f(x) = sum(square(square(x)) - 3.0 * square(x) + x)", "[1.3, -0.7, 2.0]", "[0.4, 1.5, -0.9]"},
		{"coupled", "let C = [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]\nfn f(x) = sum(sin(x) * cos(C @ x)) + sum(exp(x * 0.3))", "[0.5, -1.0, 2.0, 0.3]", "[1.0, 0.25, -0.5, 0.75]"},
		{"softmax-xent", "fn f(x) = logsumexp(x, 0) - x[2]", "[0.2, 1.1, -0.7, 0.9, 0.0]", "[0.3, -1.2, 0.8, 0.1, 2.0]"},
		{"matmul", "fn f(A) = sum(square(A @ A))", "[[1.0, 0.5], [-0.3, 0.8]]", "[[0.2, -0.4], [1.0, 0.1]]"},
	}
	for _, c := range cases {
		src := c.body + "\nlet x = " + c.x + "\nlet v = " + c.v + `
let n = numel(v)
let hv = reshape(hvp(f)(x, v), list(n))
let want = hessian(f)(x) @ reshape(v, list(n))
worst(hv, want) + worst(sum(hv * reshape(v, list(n))), fd2(f, x, v))`
		expectSmall(t, "hvp/"+c.name, src, 1e-4)
	}
}

// A scalar input is the 1-D case, where H v is just f”(x) * v, and the place
// examples/newton.tw lives.
func TestHVPOnAScalar(t *testing.T) {
	expectSmall(t, "hvp/scalar", `
fn f(t) = t * t * t * t - 3.0 * t * t + t
let x = 1.7
let v = 2.5
worst(item(hvp(f)(x, v)), (12.0 * x * x - 6.0) * v)`, 1e-12)
}

// --- refusals ---------------------------------------------------------------

func expectRefusal(t *testing.T, name, src string, want ...string) {
	t.Helper()
	_, err := runSrcErr(t, src)
	if err == nil {
		t.Fatalf("%s: was answered, not refused", name)
	}
	for _, w := range want {
		if !strings.Contains(err.Error(), w) {
			t.Errorf("%s: error %q does not mention %q", name, err.Error(), w)
		}
	}
}

func TestJVPRefusesAMismatchedTangent(t *testing.T) {
	expectRefusal(t, "wrong shape",
		"fn f(x) = sum(square(x))\nprint(jvp(f)([1.0, 2.0], [1.0, 2.0, 3.0]))",
		"jvp", "shape [3]", "shape [2]")
	expectRefusal(t, "tensor for a record",
		"fn f(p) = sum(p.w)\nprint(jvp(f)({ w: [1.0, 2.0] }, [1.0, 2.0]))",
		"jvp", "the tangent must be a record too")
	expectRefusal(t, "short list",
		"fn f(x) = sum(x[0]) + sum(x[1])\nprint(jvp(f)(list([1.0], [2.0]), list([1.0])))",
		"jvp", "list of 2", "list of 1")
	expectRefusal(t, "wrong field",
		"fn f(p) = sum(p.w)\nprint(jvp(f)({ w: [1.0] }, { b: [1.0] }))",
		"jvp", "no field w")
}

func TestVJPRefusesAMismatchedCotangent(t *testing.T) {
	expectRefusal(t, "cotangent shape",
		"fn f(x) = sum(square(x))\nprint(vjp(f)([1.0, 2.0], [1.0, 2.0]))",
		"vjp", "cotangent has shape [2]", "returned shape scalar")
}

// hvp is forward-mode over the graph, so an operation with no forward rule stops
// it just as it stops hessian, and the message has to say hvp.
func TestHVPNamesAnOperationWithoutForwardModeSupport(t *testing.T) {
	expectRefusal(t, "einsum",
		"fn f(x) = sum(einsum(\"ij,jk->ik\", x, x))\nprint(hvp(f)([[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 0.0]]))",
		"hvp", "without forward-mode support")
}

func TestHVPRefusesAMismatchedDirection(t *testing.T) {
	expectRefusal(t, "direction shape",
		"fn f(x) = sum(square(x))\nprint(hvp(f)([1.0, 2.0], [1.0, 2.0, 3.0]))",
		"hvp", "shape [3]", "shape [2]")
	expectRefusal(t, "non-scalar f",
		"fn f(x) = x * x\nprint(hvp(f)([1.0, 2.0], [1.0, 2.0]))",
		"hvp", "must return a scalar")
}

func TestArityIsEnforced(t *testing.T) {
	expectRefusal(t, "jvp one arg",
		"fn f(x) = sum(square(x))\nprint(jvp(f)([1.0, 2.0]))", "jvp")
	expectRefusal(t, "vjp three args",
		"fn f(x) = sum(square(x))\nprint(vjp(f)([1.0], [1.0], [1.0]))", "vjp")
	expectRefusal(t, "hvp one arg",
		"fn f(x) = sum(square(x))\nprint(hvp(f)([1.0, 2.0]))", "hvp")
}

// An operation with a reverse rule but no forward rule -- einsum is one -- has to
// be named as such by jvp rather than silently contributing a zero tangent. vjp
// goes the other way and works, which is the point of having both.
func TestJVPNamesAnOperationWithoutForwardModeSupport(t *testing.T) {
	expectRefusal(t, "einsum",
		"fn f(x) = sum(einsum(\"ij,jk->ik\", x, x))\nprint(jvp(f)([[1.0, 2.0], [3.0, 4.0]], [[1.0, 0.0], [0.0, 0.0]]))",
		"jvp", "without forward-mode support")
	out := runFile(t, t.TempDir(),
		"fn f(x) = sum(einsum(\"ij,jk->ik\", x, x))\nprint(vjp(f)([[1.0, 2.0], [3.0, 4.0]], 1.0)[1])\nprint(grad(f)([[1.0, 2.0], [3.0, 4.0]]))")
	if len(out) != 2 || out[0] != out[1] {
		t.Errorf("vjp and grad disagree on einsum: %v", out)
	}
}

// A derivative taken inside a derivative is the silent zero the project refuses,
// however it is spelled.
func TestNestedDerivativesAreRefused(t *testing.T) {
	expectRefusal(t, "hvp of grad",
		"fn f(x) = sum(square(x))\nprint(hvp(grad(f))([1.0, 2.0], [1.0, 0.0]))",
		"not a second derivative", "hvp(f) is already the second derivative")
	expectRefusal(t, "vjp of grad",
		"fn f(x) = sum(square(x))\nprint(vjp(grad(f))([1.0, 2.0], [1.0, 0.0]))",
		"not a second derivative")
	expectRefusal(t, "vjp inside grad",
		"fn f(x) = sum(square(x))\nfn h(x) = sum(vjp(f)(x, 1.0)[1])\nprint(grad(h)([1.0, 2.0]))",
		"vjp taken inside a derivative")
	expectRefusal(t, "grad inside jvp",
		"fn f(x) = sum(square(x))\nfn h(x) = sum(grad(f)(x))\nprint(jvp(h)([1.0, 2.0], [1.0, 0.0]))",
		"not a second derivative")
}

// A function that ignores its argument, or reaches it only through a
// piecewise-constant operation, has a zero derivative rather than an error. That
// is what grad and hessian already answer, and these must not be stricter.
func TestDerivativesOfAConstantAreZeroNotAnError(t *testing.T) {
	out := runFile(t, t.TempDir(), `fn f(x) = sum(square(floor(x)))
print(jvp(f)([1.5, 2.5], [1.0, 1.0]))
print(vjp(f)([1.5, 2.5], 1.0))
print(hvp(f)([1.5, 2.5], [1.0, 1.0]))
fn ignores(x) = 3.0
print(jvp(ignores)([1.5, 2.5], [1.0, 1.0]))
print(hvp(ignores)([1.5, 2.5], [1.0, 1.0]))
`)
	expectLines(t, out,
		"[5, 0]",
		"[5, tensor([0, 0], shape=[2])]",
		"tensor([0, 0], shape=[2])",
		"[3, 0]",
		"tensor([0, 0], shape=[2])")
}
