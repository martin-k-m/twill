package format_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/martin-k-m/twill/internal/format"
	"github.com/martin-k-m/twill/internal/parser"
)

func TestFormatExamplesRoundTrip(t *testing.T) {
	files, _ := filepath.Glob(filepath.Join("..", "..", "examples", "*.tw"))
	files2, _ := filepath.Glob(filepath.Join("..", "..", "std", "*.tw"))
	files = append(files, files2...)
	if len(files) == 0 {
		t.Fatal("no .tw files found")
	}
	for _, f := range files {
		f := f
		t.Run(filepath.Base(f), func(t *testing.T) {
			src, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			// Formatting must produce parseable output.
			out1, err := format.Source(string(src))
			if err != nil {
				t.Fatalf("format: %v", err)
			}
			if _, err := parser.Parse(out1); err != nil {
				t.Fatalf("formatted output does not re-parse: %v\n---\n%s", err, out1)
			}
			// Formatting must be idempotent.
			out2, err := format.Source(out1)
			if err != nil {
				t.Fatalf("re-format: %v", err)
			}
			if out1 != out2 {
				t.Errorf("not idempotent for %s:\n--- first ---\n%s\n--- second ---\n%s", f, out1, out2)
			}
		})
	}
}

func TestFormatPreservesPrecedence(t *testing.T) {
	// The tree must be preserved: grouping that changes associativity needs
	// parentheses in the output.
	cases := map[string]string{
		"let x = 1 + 2 * 3":         "let x = 1 + 2 * 3\n",
		"let x = (1 + 2) * 3":       "let x = (1 + 2) * 3\n",
		"let x = a - (b - c)":       "let x = a - (b - c)\n",
		"let x = a - b - c":         "let x = a - b - c\n",
		"let x = (2 ^ 3) ^ 2":       "let x = (2 ^ 3) ^ 2\n",
		"let x = 2 ^ 3 ^ 2":         "let x = 2 ^ 3 ^ 2\n",
		"let x = -(a + b)":          "let x = -(a + b)\n",
		"let x = not (a and b)":     "let x = not (a and b)\n",
		"let x = (a + b) * (c - d)": "let x = (a + b) * (c - d)\n",
	}
	for in, want := range cases {
		got, err := format.Source(in)
		if err != nil {
			t.Fatalf("%q: %v", in, err)
		}
		if got != want {
			t.Errorf("format(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestFormatBasics(t *testing.T) {
	cases := map[string]string{
		"let  x=1+2*3":                      "let x = 1 + 2 * 3\n",
		"fn f(x)=x*x":                       "fn f(x) = x * x\n",
		"fn g(A:[n,k],B:[k,m])->[n,m]{A@B}": "fn g(A: [n, k], B: [k, m]) -> [n, m] {\n  A @ B\n}\n",
		"let r={a:1,b:2}":                   "let r = { a: 1, b: 2 }\n",
		"type M={w:[2],b:[]}":               "type M = { w: [2], b: [] }\n",
	}
	for in, want := range cases {
		got, err := format.Source(in)
		if err != nil {
			t.Fatalf("%q: %v", in, err)
		}
		if got != want {
			t.Errorf("format(%q) =\n%q\nwant\n%q", in, got, want)
		}
	}
}
