package interp_test

import "testing"

// Bitwise ops on I64. `and` and `or` share the boolean keywords' spelling but
// are the bitwise builtins when called; `bnot` is bitwise complement; `shr` is
// arithmetic, so it is defined on a negative operand.

func TestBitwiseOps(t *testing.T) {
	cases := map[string]float64{
		"and(6, 3)":   2,
		"or(4, 1)":    5,
		"xor(5, 3)":   6,
		"shl(1, 4)":   16,
		"shr(0 - 8, 1)": -4, // arithmetic shift keeps the sign
		"bnot(0)":     -1,
		"and(12, or(1, 2))": 0, // 12 & 3 = 0
	}
	for src, want := range cases {
		if got := scalar(t, src); got != want {
			t.Errorf("%s = %v, want %v", src, got, want)
		}
	}
}

// The boolean keywords still work as infix operators; only a following `(`
// makes `and`/`or` a call.
func TestBooleanAndOrStillInfix(t *testing.T) {
	if got := scalar(t, "if true and false { 1.0 } else { 0.0 }"); got != 0 {
		t.Errorf("true and false took the then-branch: %v", got)
	}
	if got := scalar(t, "if 1.0 < 2.0 or false { 1.0 } else { 0.0 }"); got != 1 {
		t.Errorf("1 < 2 or false took the else-branch: %v", got)
	}
}

// A line that begins `or(` after a complete statement is a new bitwise-call
// statement, not the previous expression continued by the boolean operator.
func TestLineLeadingBitwiseCallStartsANewStatement(t *testing.T) {
	src := "mode systems\n" +
		"fn ushr(x: I64, k: I64) -> I64 {\n" +
		"  if k == 0 { return x }\n" +
		"  or(shl(x, 1), 1)\n" +
		"}\n" +
		"print(str(ushr(4, 1)))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "9" { // (4 << 1) | 1 = 9
		t.Fatalf("output = %v, want [9]", out)
	}
}
