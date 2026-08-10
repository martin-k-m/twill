package interp_test

import "testing"

// break and continue are the loop-control statements; unit is the name of the
// Unit value, so an arm like `None => unit` has a spelling.

func TestBreakAndContinue(t *testing.T) {
	src := "mode systems\n" +
		"fn f() -> F64 {\n" +
		"  let acc: F64 = 0.0\n" +
		"  let i: I64 = 0\n" +
		"  while i < 10 {\n" +
		"    i = i + 1\n" +
		"    if i == 3 { continue }\n" +
		"    if i == 7 { break }\n" +
		"    acc = acc + 1.0\n" +
		"  }\n" +
		"  acc\n" +
		"}\n" +
		"print(str(f()))\n"
	_, out := run(t, src)
	// i counts 1..7; continue skips i==3, break stops at i==7, so acc counts
	// 1, 2, 4, 5, 6 = 5.
	if len(out) != 1 || out[0] != "5" {
		t.Fatalf("output = %v, want [5]", out)
	}
}

func TestBreakInForLoop(t *testing.T) {
	src := "let sum = 0.0\n" +
		"for i in range(100) {\n" +
		"  if i == 5 { break }\n" +
		"  sum = sum + 1.0\n" +
		"}\n" +
		"print(str(sum))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "5" {
		t.Fatalf("output = %v, want [5]", out)
	}
}

func TestUnitIsAValue(t *testing.T) {
	// `unit` resolves to the Unit value; str of it is "()".
	_, out := run(t, "mode systems\nlet x = unit\nprint(str(x))\n")
	if len(out) != 1 || out[0] != "()" {
		t.Fatalf("output = %v, want [()]", out)
	}
}
