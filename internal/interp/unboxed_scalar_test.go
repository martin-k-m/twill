package interp_test

import (
	"math"
	"testing"

	"github.com/martin-k-m/raster/internal/value"
)

// A number that needs no gradient is not a tensor at runtime, so these pin the
// places where that representation has to be invisible: derivatives still see a
// graph, equality still ignores which path a value took, and everything that
// wants a tensor still gets one.

func TestGradOfAScalarLiteralArgument(t *testing.T) {
	if got := scalar(t, `grad(fn(x) { x * x })(3.0)`); got != 6 {
		t.Errorf("d/dx x^2 at 3 = %v, want 6", got)
	}
	if got := scalar(t, `grad(fn(x) { x * x * x })(2.0)`); got != 12 {
		t.Errorf("d/dx x^3 at 2 = %v, want 12", got)
	}
}

func TestHessianOfAScalarLiteralArgument(t *testing.T) {
	if got := scalar(t, `hessian(fn(x) { x * x })(2.0)`); got != 2 {
		t.Errorf("d2/dx2 x^2 at 2 = %v, want 2", got)
	}
}

func TestGradSurvivesAScalarComputedInALoop(t *testing.T) {
	// acc is built by unboxed arithmetic and then differentiated through, which
	// is the case that would break if a Num reached grad without being widened
	// back into an autodiff leaf.
	src := `
let acc = 0.0
for i in range(4) {
  acc = acc + 1.0
}
grad(fn(x) { x * acc })(3.0)
`
	if got := scalar(t, src); got != 4 {
		t.Errorf("d/dx (4x) = %v, want 4", got)
	}
}

func TestAnUnboxedNumberEqualsTheSameNumberAsATensor(t *testing.T) {
	cases := map[string]bool{
		`3.0 == scalar(3.0)`: true,
		`scalar(3.0) == 3.0`: true,
		`3.0 == 3.0`:         true,
		`3.0 == 4.0`:         false,
		// A rank-1 vector is a different shape, so it is a different value.
		`3.0 == tensor([3.0])`:     false,
		`[1.0, 2.0] == [1.0, 2.0]`: true,
	}
	for src, want := range cases {
		v, _ := run(t, src)
		got, ok := v.(value.Bool)
		if !ok {
			t.Fatalf("%s did not produce a bool, got %s", src, value.Format(v))
		}
		if bool(got) != want {
			t.Errorf("%s = %v, want %v", src, bool(got), want)
		}
	}
}

func TestScalarArithmeticMatchesTheTensorEngine(t *testing.T) {
	// Each case evaluates the same arithmetic twice: once on plain numbers,
	// once with the left side forced through a tensor. The two must agree
	// exactly, or a program could tell which path ran.
	for _, pair := range [][2]string{
		{"7.0", "3.0"},
		{"-7.0", "3.0"},
		{"7.5", "0.25"},
		{"2.0", "-1.0"},
	} {
		for _, op := range []string{"+", "-", "*", "/", "%", "^"} {
			x, y := pair[0], pair[1]
			unboxed := scalar(t, "("+x+") "+op+" ("+y+")")
			viaTensor := scalar(t, "scalar("+x+") "+op+" ("+y+")")
			if unboxed != viaTensor && !(math.IsNaN(unboxed) && math.IsNaN(viaTensor)) {
				t.Errorf("%s %s %s: unboxed %v, via tensor %v", x, op, y, unboxed, viaTensor)
			}
		}
	}
}

func TestAPlainNumberStillWorksWhereATensorIsExpected(t *testing.T) {
	if got := scalar(t, `sum(tensor([1.0, 2.0, 3.0]) * 2.0)`); got != 12 {
		t.Errorf("scalar broadcast over a tensor = %v, want 12", got)
	}
	if got := scalar(t, `sum(sqrt(4.0))`); got != 2 {
		t.Errorf("a builtin taking a tensor = %v, want 2", got)
	}
	if got := scalar(t, `len(reshape(2.0, [1]))`); got != 1 {
		t.Errorf("reshape of a plain number = %v, want 1", got)
	}
}

func TestPrintingANumberIsUnchanged(t *testing.T) {
	_, out := run(t, "print(1.0)\nprint(2.5)\nprint(-3.0)")
	want := []string{"1", "2.5", "-3"}
	if len(out) != len(want) {
		t.Fatalf("output = %q", out)
	}
	for i := range want {
		if out[i] != want[i] {
			t.Errorf("line %d = %q, want %q", i, out[i], want[i])
		}
	}
}
