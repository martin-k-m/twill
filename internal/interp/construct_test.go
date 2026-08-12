package interp_test

import (
	"strings"
	"testing"

	"github.com/martin-k-m/twill/internal/interp"
)

// linspace and arange are the tensor-construction builtins a numerical program
// reaches for first. These pin their exact output; the self-hosted parity is
// covered in selfhost_test.go.
func TestLinspaceAndArange(t *testing.T) {
	cases := map[string]string{
		// n points, both endpoints included.
		`print(linspace(0.0, 1.0, 5))`:        "tensor([0, 0.25, 0.5, 0.75, 1], shape=[5])",
		`print(linspace(2.0, 3.0, 2))`:        "tensor([2, 3], shape=[2])",
		`print(linspace(5.0, 5.0, 1))`:        "tensor([5], shape=[1])",
		`print(shape(linspace(0.0, 1.0, 0)))`: "[0]",
		// half-open, step carried, up to but not including stop.
		`print(arange(0.0, 2.0, 0.5))`: "tensor([0, 0.5, 1, 1.5], shape=[4])",
		`print(arange(0.0, 1.0, 1.0))`: "tensor([0], shape=[1])",
		// an empty range when stop is not past start.
		`print(shape(arange(5.0, 5.0, 1.0)))`: "[0]",
	}
	for src, want := range cases {
		_, out := run(t, src+"\n")
		got := strings.TrimSpace(strings.Join(out, ""))
		if got != want {
			t.Errorf("%s\n  got  %q\n  want %q", src, got, want)
		}
	}
}

func TestArangeRejectsZeroStep(t *testing.T) {
	ip := interp.New(func(string) {})
	if _, err := ip.Run("arange(0.0, 1.0, 0.0)\n"); err == nil {
		t.Fatal("expected an error for a zero step")
	}
}

// arr_push is the arr_new/arr_clear family's append, the same as push. It is
// what several systems-mode callers reach for.
func TestArrPushAppendsLikePush(t *testing.T) {
	_, out := run(t, "let a = arr_new()\narr_push(a, 1)\narr_push(a, 2)\narr_push(a, 3)\nprint(a)\nprint(len(a))\n")
	if got := strings.Join(out, "|"); got != "[1, 2, 3]|3" {
		t.Fatalf("got %q, want %q", got, "[1, 2, 3]|3")
	}
}
