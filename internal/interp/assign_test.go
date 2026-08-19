package interp_test

import (
	"strings"
	"testing"
)

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

// `m[i][j] = v` on a tensor of rank 2 or more used to write nothing at all:
// indexing a row hands back a copy, so the assignment went into the copy and
// vanished, with no change and no error. A list of lists was never affected,
// because a list is a handle and the inner one is shared -- which is why this
// was only ever wrong for the tensors numerical code is made of.
func TestNestedTensorAssignment(t *testing.T) {
	out := runFile(t, t.TempDir(), `let m = zeros(2, 3)
m[0][1] = 5.0
m[1][2] = 9.0
print(m)
let c = zeros(2, 2, 2)
c[1][0][1] = 4.0
print(c)
let r = { w: zeros(2, 2) }
r.w[1][0] = 3.0
print(r.w)
let xs = list(list(1.0, 2.0), list(3.0, 4.0))
xs[0][1] = 8.0
print(xs)
let v = zeros(3)
v[1] = 7.0
print(v)
`)
	expectLines(t, out,
		"tensor([[0, 5, 0], [0, 0, 9]], shape=[2, 3])",
		"tensor([[[0, 0], [0, 0]], [[0, 4], [0, 0]]], shape=[2, 2, 2])",
		"tensor([[0, 0], [3, 0]], shape=[2, 2])",
		"[[1, 8], [3, 4]]",
		"tensor([0, 7, 0], shape=[3])")
}

// Each axis is bounds-checked on its own, and a partial index is refused
// rather than quietly writing somewhere.
func TestNestedTensorAssignmentErrors(t *testing.T) {
	for _, tc := range []struct{ src, want string }{
		{"let m = zeros(2, 3)\nm[5][0] = 1.0\n", "index 5 out of range for axis 0 of length 2"},
		{"let m = zeros(2, 3)\nm[0][9] = 1.0\n", "index 9 out of range for axis 1 of length 3"},
		{"let m = zeros(2, 2, 2)\nm[0][1] = 1.0\n", "assigning to a row needs 3 indices, got 2"},
	} {
		_, err := runSrcErr(t, tc.src)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("err = %v, want %q", err, tc.want)
		}
	}
}
