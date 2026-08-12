package interp_test

import (
	"strings"
	"testing"
)

// sort on a list of strings uses the bytewise-unsigned lexicographic order
// docs/language-guide.md pins (NEEDS-23): Go's own string comparison. These pin
// the exact result, not just that it agrees with the self-hosted evaluator.
func TestSortStringListOrdering(t *testing.T) {
	cases := map[string]string{
		// Uppercase 'A' (65) sorts before lowercase 'a' (97).
		`print(sort(["banana", "apple", "cherry", "Apple"]))`: "[Apple, apple, banana, cherry]",
		// A shared prefix: the shorter string sorts first.
		`print(sort(["ab", "a", "abc"]))`: "[a, ab, abc]",
		// The descending flag reverses the ascending order.
		`print(sort(["b", "a", "c"], true))`: "[c, b, a]",
		// A single-element and already-sorted list are returned as-is.
		`print(sort(["only"]))`: "[only]",
	}
	for src, want := range cases {
		_, out := run(t, src+"\n")
		got := strings.TrimSpace(strings.Join(out, ""))
		if got != want {
			t.Errorf("%s\n  got  %q\n  want %q", src, got, want)
		}
	}
}

func TestSortStringListDoesNotMutateInput(t *testing.T) {
	// The returned list is fresh; the original binding keeps its order, the same
	// contract the tensor sort has.
	_, out := run(t, "let xs = [\"b\", \"a\"]\nlet ys = sort(xs)\nprint(xs)\nprint(ys)\n")
	got := strings.Join(out, "|")
	if want := "[b, a]|[a, b]"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
