package interp_test

import "testing"

// Assignment targets an lvalue: a name, a field (`obj.f = v`), or an index
// (`arr[i] = v`), and these compose. Records and lists are reference values, so
// a field or element write mutates the object every binding shares.

func TestFieldAssignmentMutatesTheRecord(t *testing.T) {
	src := "let r = { n: 5.0 }\n" +
		"r.n = r.n + 7.0\n" +
		"print(str(r.n))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "12" {
		t.Fatalf("output = %v, want [12]", out)
	}
}

func TestIndexAssignmentMutatesAList(t *testing.T) {
	src := "let xs = [10.0, 20.0, 30.0]\n" +
		"xs[1] = 99.0\n" +
		"print(str(xs[1]))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "99" {
		t.Fatalf("output = %v, want [99]", out)
	}
}

// A field write is visible through an aliasing binding, proving records are
// references rather than copies.
func TestFieldWriteIsVisibleThroughAnAlias(t *testing.T) {
	src := "let a = { v: 1.0 }\n" +
		"let b = a\n" +
		"b.v = 42.0\n" +
		"print(str(a.v))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "42" {
		t.Fatalf("output = %v, want [42] (record is a reference)", out)
	}
}

// A field of an indexed element: `xs[0].n = v`, the composing case the systems
// sources use (`a.d[i] = v`).
func TestFieldOfIndexedElement(t *testing.T) {
	src := "let xs = [{ n: 1.0 }, { n: 2.0 }]\n" +
		"xs[0].n = 5.0\n" +
		"print(str(xs[0].n))\n"
	_, out := run(t, src)
	if len(out) != 1 || out[0] != "5" {
		t.Fatalf("output = %v, want [5]", out)
	}
}
