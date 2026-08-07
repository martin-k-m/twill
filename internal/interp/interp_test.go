package interp_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/martin-k-m/twill/internal/interp"
	"github.com/martin-k-m/twill/internal/value"
)

func run(t *testing.T, src string) (value.Value, []string) {
	t.Helper()
	var out []string
	ip := interp.New(func(s string) { out = append(out, s) })
	v, err := ip.Run(src)
	if err != nil {
		t.Fatalf("run error: %v\nsource:\n%s", err, src)
	}
	return v, out
}

// scalar reads the number a program produced, without caring whether it came
// back unboxed or as a rank-0 tensor. Which one it is is an interpreter detail
// and no test should be pinning it.
func scalar(t *testing.T, src string) float64 {
	t.Helper()
	v, _ := run(t, src)
	n, ok := value.AsNumber(v)
	if !ok {
		t.Fatalf("expected a number, got %s", value.Format(v))
	}
	return n
}

func TestArithmeticPrecedence(t *testing.T) {
	cases := map[string]float64{
		"1 + 2 * 3":   7,
		"(1 + 2) * 3": 9,
		"2 ^ 10":      1024,
		"17 % 5":      2,
		"-3 + 4":      1,
	}
	for src, want := range cases {
		if got := scalar(t, src); got != want {
			t.Errorf("%q = %v, want %v", src, got, want)
		}
	}
}

func TestFunctionsAndClosures(t *testing.T) {
	if got := scalar(t, "let a = 10\nfn double(x) = x * 2\ndouble(a) + 1"); got != 21 {
		t.Errorf("got %v", got)
	}
	src := "fn adder(n) = fn(x) = x + n\nlet add5 = adder(5)\nadd5(37)"
	if got := scalar(t, src); got != 42 {
		t.Errorf("got %v", got)
	}
}

func TestControlFlow(t *testing.T) {
	if got := scalar(t, "if 3 > 2 { 100 } else { 200 }"); got != 100 {
		t.Errorf("if got %v", got)
	}
	while := "let i = 0\nlet total = 0\nwhile i < 5 { total = total + i\ni = i + 1 }\ntotal"
	if got := scalar(t, while); got != 10 {
		t.Errorf("while got %v", got)
	}
	forsrc := "let s = 0\nfor k in range(1, 6) { s = s + k }\ns"
	if got := scalar(t, forsrc); got != 15 {
		t.Errorf("for got %v", got)
	}
}

func TestTensorsMatmulIndex(t *testing.T) {
	if got := scalar(t, "[1.0, 2.0, 3.0] @ [4.0, 5.0, 6.0]"); got != 32 {
		t.Errorf("dot got %v", got)
	}
	if got := scalar(t, "let m = [[1.0, 2.0], [3.0, 4.0]]\nm[1][0]"); got != 3 {
		t.Errorf("index got %v", got)
	}
	if got := scalar(t, "sum([1.0, 2.0, 3.0, 4.0])"); got != 10 {
		t.Errorf("sum got %v", got)
	}
}

func TestGrad(t *testing.T) {
	if got := scalar(t, "grad(fn(x) = x * x * x)(2.0)"); got != 12 {
		t.Errorf("grad cube got %v", got)
	}
	if got := scalar(t, "grad(fn(x) = sum(x * x))([3.0, 4.0])[0]"); got != 6 {
		t.Errorf("grad vec got %v", got)
	}
}

func TestGradsPerArgument(t *testing.T) {
	src := "fn bil(a, b) = sum(a * b)\nlet g = grads(bil)([1.0, 2.0], [10.0, 20.0])\ng[0][1]"
	if got := scalar(t, src); got != 20 {
		t.Errorf("grads got %v", got)
	}
}

func TestGradientDescent(t *testing.T) {
	src := `
		let w = 0.0
		fn loss(w) = (w - 3.0) * (w - 3.0)
		for step in range(200) {
			let g = grad(loss)(w)
			w = w - g * 0.1
		}
		w`
	if got := scalar(t, src); math.Abs(got-3) > 1e-3 {
		t.Errorf("descent got %v, want ~3", got)
	}
}

func TestMapZip(t *testing.T) {
	src := "let xs = map(fn(x) = x * x, [1.0, 2.0, 3.0])\nxs[2]"
	if got := scalar(t, src); got != 9 {
		t.Errorf("map got %v", got)
	}
	zipsrc := "let z = zip([1.0, 2.0], [10.0, 20.0])\nz[1][0] + z[1][1]"
	if got := scalar(t, zipsrc); got != 22 {
		t.Errorf("zip got %v", got)
	}
}

func TestLeadingMinusStartsNewStatement(t *testing.T) {
	// The final line begins with '-', which must be its own statement (the
	// block's return value), not a subtraction glued to `let b`.
	src := `
		fn f(x) {
			let b = x * 2.0
			-b
		}
		f(5.0)`
	if got := scalar(t, src); got != -10 {
		t.Errorf("got %v, want -10", got)
	}
}

func TestPrintFormatting(t *testing.T) {
	_, out := run(t, `print("v =", [1.0, 2.0])`)
	if len(out) != 1 || out[0] != "v = tensor([1, 2], shape=[2])" {
		t.Errorf("got %q", out)
	}
}

func TestImport(t *testing.T) {
	dir := t.TempDir()
	lib := filepath.Join(dir, "lib.tw")
	if err := os.WriteFile(lib, []byte("fn triple(x) = x * 3.0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	main := filepath.Join(dir, "main.tw")
	body := "import \"lib.tw\"\nprint(triple(4.0))\n"
	if err := os.WriteFile(main, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	var out []string
	ip := interp.New(func(s string) { out = append(out, s) })
	if err := ip.RunFile(main); err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0] != "12" {
		t.Errorf("import output = %q", out)
	}
}
