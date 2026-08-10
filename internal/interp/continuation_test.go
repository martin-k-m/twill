package interp_test

import "testing"

// A line that begins with `+`/`-` ends the previous statement at statement level,
// but inside a grouping (a call's arguments, a parenthesised expression, a list)
// there is no statement to end, so the operator continues the expression.

func TestLeadingOperatorContinuesInsideCallArgs(t *testing.T) {
	// f(1 + 2, 3) written with the argument split across a line-leading `+`.
	src := "fn f(a, b) = a + b\n" +
		"let r = f(1.0\n  + 2.0, 3.0)\n" +
		"print(str(r))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "6" {
		t.Fatalf("output = %v, want [6]", out)
	}
}

func TestLeadingOperatorContinuesInsideParens(t *testing.T) {
	src := "let r = (10.0\n  - 3.0)\n" +
		"print(str(r))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "7" {
		t.Fatalf("output = %v, want [7]", out)
	}
}

// At statement level the rule still holds: a line-leading `-` starts a new
// statement rather than subtracting from the line above.
func TestLeadingOperatorStillBreaksAtStatementLevel(t *testing.T) {
	src := "let m = 2.0\n" +
		"let x = 5.0\n" +
		"- m\n" +
		"print(str(x))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "5" {
		t.Fatalf("output = %v, want [5] (the `- m` line is its own statement)", out)
	}
}
