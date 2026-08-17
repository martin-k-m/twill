package interp_test

import (
	"testing"

	"github.com/twill-lang/twill/internal/format"
)

// enum declares a sum type; each case is a value in scope (a constructor when it
// carries a payload, the variant itself when it does not). match inspects the
// case, binds its payload, and runs the arm.

func TestEnumMatchExpressionArms(t *testing.T) {
	src := "mode systems\n" +
		"enum Opt { Some(F64), None }\n" +
		"fn unwrap_or(o: Opt, d: F64) -> F64 = match o { Some(v) => v, None => d }\n" +
		"print(str(unwrap_or(Some(42.0), 0.0)))\n" +
		"print(str(unwrap_or(None, 7.0)))\n"
	_, out := run(t, src)
	if len(out) != 2 || out[0] != "42" || out[1] != "7" {
		t.Fatalf("output = %v, want [42 7]", out)
	}
}

func TestMatchStatementArms(t *testing.T) {
	// An assignment arm and a return arm, the shapes the compiler's own sources
	// use.
	src := "mode systems\n" +
		"enum Res { Ok(F64), Err(Str) }\n" +
		"fn get(r: Res) -> F64 {\n" +
		"  let acc: F64 = 0.0\n" +
		"  match r {\n" +
		"    Ok(v) => acc = v,\n" +
		"    Err(_) => return 0.0 - 1.0,\n" +
		"  }\n" +
		"  acc\n" +
		"}\n" +
		"print(str(get(Ok(5.0))))\n" +
		"print(str(get(Err(\"bad\"))))\n"
	_, out := run(t, src)
	if len(out) != 2 || out[0] != "5" || out[1] != "-1" {
		t.Fatalf("output = %v, want [5 -1]", out)
	}
}

func TestMatchWildcardArm(t *testing.T) {
	src := "mode systems\n" +
		"enum Color { Red, Green, Blue }\n" +
		"fn code(c: Color) -> F64 = match c { Red => 1.0, _ => 0.0 }\n" +
		"print(str(code(Red)))\n" +
		"print(str(code(Blue)))\n"
	_, out := run(t, src)
	if len(out) != 2 || out[0] != "1" || out[1] != "0" {
		t.Fatalf("output = %v, want [1 0]", out)
	}
}

func TestEnumAndMatchRoundTripThroughFmt(t *testing.T) {
	src := "enum Opt { Some(F64), None }\n" +
		"fn f(o: Opt, d: F64) -> F64 = match o { Some(v) => v, None => d }\n"
	got, err := format.Source(src)
	if err != nil {
		t.Fatalf("format error: %v", err)
	}
	if got != src {
		t.Fatalf("round-trip changed the source:\n got: %q\nwant: %q", got, src)
	}
}
