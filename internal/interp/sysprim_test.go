package interp_test

import (
	"strings"
	"testing"

	"github.com/martin-k-m/twill/internal/interp"
)

// runOut runs a program capturing everything it prints, joined by newlines.
func runOut(t *testing.T, src string) string {
	t.Helper()
	var out []string
	ip := interp.New(func(s string) { out = append(out, s) })
	if _, err := ip.Run(src); err != nil {
		t.Fatalf("run: %v", err)
	}
	return strings.Join(out, "\n")
}

// is_same is reference identity: two bindings of one list are the same object, a
// fresh list is not, and a mutation through one alias is seen through the other.
func TestIsSameIsReferenceIdentity(t *testing.T) {
	got := runOut(t, "let a = arr_new()\nlet b = a\nprint(is_same(a, b))\nprint(is_same(a, arr_new()))")
	if got != "true\nfalse" {
		t.Fatalf("is_same = %q, want \"true\\nfalse\"", got)
	}
}

// emit_line writes its argument and a newline, so two calls print two lines.
func TestEmitLine(t *testing.T) {
	got := runOut(t, "emit_line(\"one\")\nemit_line(\"two\")")
	if got != "one\ntwo" {
		t.Fatalf("emit_line = %q, want \"one\\ntwo\"", got)
	}
}

// A seed makes the scalar generator reproducible: the same seed yields the same
// first draw, so a run is repeatable.
func TestRngSeedIsReproducible(t *testing.T) {
	first := runOut(t, "rng_seed(7)\nprint(rng_uniform())")
	second := runOut(t, "rng_seed(7)\nprint(rng_uniform())")
	if first != second {
		t.Fatalf("seeded draws differ: %q vs %q", first, second)
	}
}

// rng_perm returns a permutation of 0..n-1, so its length is n.
func TestRngPermLength(t *testing.T) {
	got := runOut(t, "rng_seed(1)\nprint(len(rng_perm(5)))")
	if got != "5" {
		t.Fatalf("len(rng_perm(5)) = %q, want \"5\"", got)
	}
}
