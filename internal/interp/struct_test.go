package interp_test

import (
	"testing"

	"github.com/martin-k-m/twill/internal/format"
)

// `struct Name { field: Type, ... }` declares a record type. Records are
// structural, so the declaration is erased at runtime and a value is built with
// a (typed) record literal; the checker registers the field names so `m.field`
// is checked for existence.

func TestStructDeclAndFieldAccess(t *testing.T) {
	src := "mode systems\n" +
		"struct Mat { rows: I64, cols: I64, data: Arr[F64] }\n" +
		"fn area(m: Mat) -> I64 = m.rows * m.cols\n" +
		"let m = Mat { rows: 3.0, cols: 4.0, data: [1.0] }\n" +
		"print(str(area(m)))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "12" {
		t.Fatalf("output = %v, want [12]", out)
	}
}

func TestStructWithGenericAndQualifiedFieldTypesParses(t *testing.T) {
	src := "struct Env { vars: Dict[Str, I64], node: ast.Expr, xs: Arr[Arr[F64]] }\n"
	got, err := format.Source(src)
	if err != nil {
		t.Fatalf("format error: %v", err)
	}
	if got != src {
		t.Fatalf("round-trip changed the source:\n got: %q\nwant: %q", got, src)
	}
}
