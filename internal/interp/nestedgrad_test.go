package interp_test

import (
	"strings"
	"testing"
)

// Reverse mode is first-order: the gradient it hands back is a plain value with
// no history, so differentiating it again differentiates a constant and the
// answer is zero. `grad(grad(f))` was already refused. This is the same mistake
// written the long way round -- a user function between the two gradients --
// which the direct check could not see, and which returned the silent zero that
// is the worst way for a derivative to be wrong.
func TestGradientInsideAGradientIsRefused(t *testing.T) {
	_, err := runSrcErr(t, `fn f(x) = sum(x * x * x)
let g = grad(f)
fn h(x) = sum(g(x)) * 2.0
print(grad(h)([3.0]))
`)
	if err == nil {
		t.Fatal("a gradient taken inside a gradient was answered, not refused")
	}
	for _, want := range []string{"not a second derivative", "hessian(f)", "jacobian(f)"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err.Error(), want)
		}
	}
}

// The nesting hessian and jacobian do themselves is not the refused one: they
// run forward mode over the reverse pass, or one reverse pass per output, and
// both are correct. Refusing them would be refusing the answer.
func TestHessianAndJacobianStillNest(t *testing.T) {
	out := runFile(t, t.TempDir(), `fn f(v) = (v[0] - v[1]) * (v[0] - v[1]) + v[1] * v[1]
print(hessian(f)([1.0, 2.0]))
fn g(v) {
  let a = v[0:1]
  let b = v[1:2]
  concat([a * b, sin(a)], 0)
}
print(jacobian(g)([1.0, 0.5]))
fn loss(m) = sum(m.w * m.w)
print(grad(loss)({ w: [2.0, 3.0] }))
`)
	expectLines(t, out,
		"tensor([[2, -2], [-2, 4]], shape=[2, 2])",
		"tensor([[0.5, 1], [0.540302, 0]], shape=[2, 2])",
		"{w: tensor([4, 6], shape=[2])}")
}

// A gradient taken after another has finished is ordinary code and stays so:
// the depth counter has to unwind.
func TestSequentialGradientsAreFine(t *testing.T) {
	out := runFile(t, t.TempDir(), `fn f(x) = sum(x * x)
let a = grad(f)([1.0, 2.0])
let b = grad(f)([3.0, 4.0])
print(a, b)
fn twice(x) = sum(grad(f)(x))
print(twice([1.0, 2.0]))
`)
	expectLines(t, out,
		"tensor([2, 4], shape=[2]) tensor([6, 8], shape=[2])",
		"6")
}
