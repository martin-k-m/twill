package interp_test

import (
	"testing"

	"github.com/martin-k-m/twill/internal/format"
)

// A typed record literal, `Point { x: 1.0 }`, builds the same structural record
// as `{ x: 1.0 }`; the type name is advisory. looksLikeRecord requires
// `{ ident :`, so an `if`/`while` block whose condition ends in a name is not
// mistaken for one.

func TestTypedRecordLiteralBuildsAStructuralRecord(t *testing.T) {
	src := "mode systems\n" +
		"let p = Point { x: 3.0, y: 4.0 }\n" +
		"print(str(p.x))\n" +
		"print(str(p.y))\n"
	_, out := run(t, src)
	if len(out) != 2 || out[0] != "3" || out[1] != "4" {
		t.Fatalf("output = %v, want [3 4]", out)
	}
}

func TestTypedRecordFromAFunctionWorksLikeAnonymous(t *testing.T) {
	src := "mode systems\n" +
		"fn mk(b: F64) -> Point = Point { x: b, y: b }\n" +
		"print(str(mk(9.0).y))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "9" {
		t.Fatalf("output = %v, want [9]", out)
	}
}

func TestRecordLiteralDoesNotSwallowAnIfBlock(t *testing.T) {
	src := "mode systems\n" +
		"let p = { x: 1.0 }\n" +
		"if p.x > 0.0 { print(\"pos\") }\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "pos" {
		t.Fatalf("output = %v, want [pos]", out)
	}
}

func TestTypedRecordRoundTripsThroughFmt(t *testing.T) {
	src := "let p = Point { x: 3.5, y: 4.5 }\n" +
		"let q = geom.Point { x: 1.5 }\n"
	got, err := format.Source(src)
	if err != nil {
		t.Fatalf("format error: %v", err)
	}
	if got != src {
		t.Fatalf("round-trip changed the source:\n got: %q\nwant: %q", got, src)
	}
}
