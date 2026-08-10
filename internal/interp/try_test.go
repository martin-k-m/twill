package interp_test

import (
	"testing"

	"github.com/martin-k-m/twill/internal/format"
)

// Postfix `?` unwraps the success case of a Res/Opt (the payload of `Ok`/`Some`)
// or returns its failure case (`Err`/`None`) from the enclosing function. Ok,
// Err, Some and None are built in, so no declaration is needed.

func TestTryUnwrapsAndPropagates(t *testing.T) {
	src := "mode systems\n" +
		"fn half(x: F64) -> Res = if x > 0.0 { Ok(x / 2.0) } else { Err(\"neg\") }\n" +
		"fn run(x: F64) -> Res {\n" +
		"  let h: F64 = half(x)?\n" +
		"  Ok(h + 1.0)\n" +
		"}\n" +
		"print(str(match run(10.0) { Ok(v) => v, Err(_) => 0.0 - 1.0 }))\n" +
		"print(str(match run(0.0 - 4.0) { Ok(v) => v, Err(_) => 0.0 - 1.0 }))\n"
	_, out := run(t, src)
	if len(out) != 2 || out[0] != "6" || out[1] != "-1" {
		t.Fatalf("output = %v, want [6 -1] (Ok unwraps, Err propagates)", out)
	}
}

func TestOptSomeAndNone(t *testing.T) {
	src := "mode systems\n" +
		"fn first(present: F64) -> Opt = if present > 0.0 { Some(9.0) } else { None }\n" +
		"fn get(present: F64) -> Opt {\n" +
		"  let v: F64 = first(present)?\n" +
		"  Some(v + 1.0)\n" +
		"}\n" +
		"print(str(match get(1.0) { Some(v) => v, None => 0.0 - 1.0 }))\n" +
		"print(str(match get(0.0) { Some(v) => v, None => 0.0 - 1.0 }))\n"
	_, out := run(t, src)
	if len(out) != 2 || out[0] != "10" || out[1] != "-1" {
		t.Fatalf("output = %v, want [10 -1]", out)
	}
}

func TestTryRoundTripsThroughFmt(t *testing.T) {
	src := "fn run(x: F64) -> Res = Ok(half(x)?)\n"
	got, err := format.Source(src)
	if err != nil {
		t.Fatalf("format error: %v", err)
	}
	if got != src {
		t.Fatalf("round-trip changed the source:\n got: %q\nwant: %q", got, src)
	}
}
