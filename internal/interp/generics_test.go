package interp_test

import (
	"testing"

	"github.com/twill-lang/twill/internal/ast"
	"github.com/twill-lang/twill/internal/format"
	"github.com/twill-lang/twill/internal/parser"
)

// Generic type names (`Arr[I64]`, `Dict[Str, V]`, `Res[T, E]`) appear on
// parameters, returns and lets throughout the systems-mode sources. The type
// grammar reuses the unit-expression parser, which stops at the `[`; a `[` after
// a bare name is a generic argument list, kept as advisory text since the
// bootstrap has no such type.

func TestGenericParamAndReturnParse(t *testing.T) {
	fn := fnDecl(t, "fn first(xs: Arr[I64]) -> Res[I64, Str] = xs\n")
	if got := fn.Params[0].TypeName; got != "Arr[I64]" {
		t.Fatalf("param type = %q, want %q", got, "Arr[I64]")
	}
	if got := fn.RetType; got != "Res[I64, Str]" {
		t.Fatalf("return type = %q, want %q", got, "Res[I64, Str]")
	}
}

func TestNestedAndQualifiedGenericArgsParse(t *testing.T) {
	fn := fnDecl(t, "fn f(a: Arr[Arr[I64]], b: Dict[Str, ast.Expr]) = a\n")
	if got := fn.Params[0].TypeName; got != "Arr[Arr[I64]]" {
		t.Fatalf("param 0 = %q, want %q", got, "Arr[Arr[I64]]")
	}
	if got := fn.Params[1].TypeName; got != "Dict[Str, ast.Expr]" {
		t.Fatalf("param 1 = %q, want %q", got, "Dict[Str, ast.Expr]")
	}
}

func TestGenericLetAnnotationParses(t *testing.T) {
	prog, err := parser.Parse("let d: Arr[Arr[I64]] = 3\n")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	let, ok := prog.Body[0].(*ast.Let)
	if !ok {
		t.Fatalf("first statement is %T, want *ast.Let", prog.Body[0])
	}
	if let.TypeName != "Arr[Arr[I64]]" {
		t.Fatalf("let type = %q, want %q", let.TypeName, "Arr[Arr[I64]]")
	}
	if let.Unit != nil {
		t.Fatalf("a generic annotation is a type, not a unit; Unit should be nil")
	}
}

func TestFmtRoundTripsGenerics(t *testing.T) {
	src := "fn f(a: Arr[I64], b: Dict[Str, Arr[F64]]) -> Res[I64, Str] = a\n" +
		"let d: Arr[Arr[I64]] = 3\n"
	got, err := format.Source(src)
	if err != nil {
		t.Fatalf("format error: %v", err)
	}
	if got != src {
		t.Fatalf("round-trip changed the source:\n got: %q\nwant: %q", got, src)
	}
}
