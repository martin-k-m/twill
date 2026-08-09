package interp_test

import (
	"testing"

	"github.com/martin-k-m/twill/internal/ast"
	"github.com/martin-k-m/twill/internal/format"
	"github.com/martin-k-m/twill/internal/parser"
)

// A module-qualified type name (`cp.Caps`) is the pervasive systems-mode
// convention for a parameter or return type. The type grammar reuses the
// unit-expression parser, which stops at the `.`; the qualified suffix is lifted
// out into an advisory type name. Units are never qualified, so a `.` is an
// unambiguous marker that the name is a type.

func fnDecl(t *testing.T, src string) *ast.FnDecl {
	t.Helper()
	prog, err := parser.Parse(src)
	if err != nil {
		t.Fatalf("parse error: %v\nsource: %s", err, src)
	}
	fn, ok := prog.Body[0].(*ast.FnDecl)
	if !ok {
		t.Fatalf("first statement is %T, want *ast.FnDecl", prog.Body[0])
	}
	return fn
}

func TestQualifiedParamTypeParses(t *testing.T) {
	fn := fnDecl(t, "fn f(c: cp.Caps) = c\n")
	if got := fn.Params[0].TypeName; got != "cp.Caps" {
		t.Fatalf("param type = %q, want %q", got, "cp.Caps")
	}
}

func TestQualifiedReturnTypeParses(t *testing.T) {
	fn := fnDecl(t, "fn f(c: cp.Caps) -> cp.Caps = c\n")
	if got := fn.RetType; got != "cp.Caps" {
		t.Fatalf("return type = %q, want %q", got, "cp.Caps")
	}
	if fn.RetUnit != nil {
		t.Fatalf("a qualified return is a type, not a unit; RetUnit should be nil")
	}
}

func TestDeeplyQualifiedTypeParses(t *testing.T) {
	fn := fnDecl(t, "fn f(x: a.b.c) -> a.b.c = x\n")
	if got := fn.Params[0].TypeName; got != "a.b.c" {
		t.Fatalf("param type = %q, want %q", got, "a.b.c")
	}
	if got := fn.RetType; got != "a.b.c" {
		t.Fatalf("return type = %q, want %q", got, "a.b.c")
	}
}

// A bare name after `->` is left on the existing path (a unit, or a bare type
// the checker resolves), unchanged by this feature. Only a `.` diverts it.
func TestBareReturnNameIsUnchanged(t *testing.T) {
	fn := fnDecl(t, "fn price() -> USD = 1.0\n")
	if fn.RetType != "" {
		t.Fatalf("a bare `-> USD` is not a qualified type; RetType = %q, want empty", fn.RetType)
	}
	if fn.RetUnit == nil {
		t.Fatalf("a bare `-> USD` should still be a unit return")
	}
}

func TestQualifiedTypeIsAdvisoryAndDoesNotBlockChecking(t *testing.T) {
	// cp is not imported and cp.Caps is not a declared type; the annotation is
	// advisory, so this checks clean and runs, returning its argument.
	_, out := run(t, "mode systems\nfn idc(c: cp.Caps) -> cp.Caps = c\nprint(str(idc(42.0)))\n")
	if len(out) != 1 || out[0] != "42" {
		t.Fatalf("output = %v, want [42]", out)
	}
}

func TestFmtRoundTripsQualifiedTypes(t *testing.T) {
	src := "fn make(c: cp.Caps, by: cp.Rgb) -> cp.Caps = c\n"
	got, err := format.Source(src)
	if err != nil {
		t.Fatalf("format error: %v", err)
	}
	if got != src {
		t.Fatalf("round-trip changed the source:\n got: %q\nwant: %q", got, src)
	}
}
