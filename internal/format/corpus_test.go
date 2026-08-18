package format_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/twill-lang/twill/internal/format"
	"github.com/twill-lang/twill/internal/parser"
)

// The formatter's contract, over every .tw file in the repository rather than
// the handful the round-trip test uses: the corpus fixtures, the self-hosted
// compiler, the standard library and the examples, which between them are some
// seven hundred files and every construct the language has.
//
// Three properties, and the third is the one worth the file:
//
//   - what it prints parses;
//   - formatting twice is formatting once (fmt(fmt(x)) == fmt(x));
//   - every comment that went in comes out, and so does every statement.
//
// A formatter that drops something is worse than one that refuses, because the
// refusal is visible and the loss is not: `twill fmt --write` overwrites the
// file, and the comment explaining why a constant is 37 is gone with no
// diagnostic anywhere. format.Source is documented as refusing rather than
// moving a comment it cannot place, and this is what holds it to that.
//
// The statement count is here because a comment check would not have caught
// what was actually wrong: `unit USD` had no case in the printer at all, so the
// declaration was silently deleted and every annotation naming it then failed
// to check (NEEDS-77). Counting what goes in against what comes out is the
// property that catches a whole missing case rather than a misplaced one.

func corpusFiles(t *testing.T) []string {
	t.Helper()
	var files []string
	for _, pat := range []string{
		filepath.Join("..", "..", "testdata", "cases", "*.tw"),
		filepath.Join("..", "..", "testdata", "examples", "*.tw"),
		filepath.Join("..", "..", "src", "*.tw"),
		filepath.Join("..", "..", "src", "cli", "*.tw"),
		filepath.Join("..", "..", "src", "gpu", "*.tw"),
		filepath.Join("..", "..", "std", "*.tw"),
		filepath.Join("..", "..", "std", "term", "*.tw"),
		filepath.Join("..", "..", "std", "tests", "*.tw"),
		filepath.Join("..", "..", "examples", "*.tw"),
		filepath.Join("..", "..", "bench", "workloads", "*.tw"),
	} {
		found, _ := filepath.Glob(pat)
		files = append(files, found...)
	}
	if len(files) < 100 {
		t.Fatalf("corpus is %d files, which is too few to be the corpus", len(files))
	}
	return files
}

func TestFormatterOverTheWholeCorpus(t *testing.T) {
	files := corpusFiles(t)
	var refused int
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		// A fixture that does not parse is a fixture about a parse error, and
		// the formatter is right to refuse it.
		if _, perr := parser.Parse(string(src)); perr != nil {
			continue
		}
		out, err := format.Source(string(src))
		if err != nil {
			// A refusal is allowed and is the documented behaviour for a comment
			// that cannot be placed. It is counted so that a formatter which
			// starts refusing everything cannot pass this test quietly.
			refused++
			continue
		}
		if _, err := parser.Parse(out); err != nil {
			t.Errorf("%s: formatted output does not parse: %v", f, err)
			continue
		}
		again, err := format.Source(out)
		if err != nil {
			t.Errorf("%s: formatted output does not format: %v", f, err)
			continue
		}
		if again != out {
			t.Errorf("%s: not idempotent\n--- once ---\n%s\n--- twice ---\n%s",
				f, firstDifference(out, again), firstDifference(again, out))
		}
		if missing := lostComments(string(src), out); len(missing) > 0 {
			t.Errorf("%s: formatting dropped %d comment(s), first: %q", f, len(missing), missing[0])
		}
		before, _ := parser.Parse(string(src))
		after, err := parser.Parse(out)
		if err == nil && len(before.Body) != len(after.Body) {
			t.Errorf("%s: formatting changed the statement count, %d in and %d out",
				f, len(before.Body), len(after.Body))
		}
	}
	if refused*4 > len(files) {
		t.Errorf("the formatter refused %d of %d files, which is not a formatter", refused, len(files))
	}
	t.Logf("%d files, %d refused", len(files), refused)
}

// lostComments reports the comment bodies in src that are not in out. It
// compares the text of each comment rather than counting them, so a comment
// that was replaced by a different one is caught and one that merely moved is
// not: where a comment lands is the formatter's business, whether it survives
// is not.
//
// A comment marker inside a string literal is not a comment, so lines are
// scanned with quoting tracked, the way the lexer does it.
func lostComments(src, out string) []string {
	have := map[string]int{}
	for _, c := range commentsIn(out) {
		have[c]++
	}
	var missing []string
	for _, c := range commentsIn(src) {
		if have[c] > 0 {
			have[c]--
			continue
		}
		missing = append(missing, c)
	}
	return missing
}

func commentsIn(src string) []string {
	var out []string
	for _, line := range strings.Split(src, "\n") {
		inString := false
		escaped := false
		for i := 0; i < len(line); i++ {
			ch := line[i]
			switch {
			case escaped:
				escaped = false
			case ch == '\\' && inString:
				escaped = true
			case ch == '"':
				inString = !inString
			case ch == '#' && !inString:
				out = append(out, strings.TrimSpace(line[i+1:]))
				i = len(line)
			}
		}
	}
	return out
}

// firstDifference trims a rendering down to the neighbourhood of where two
// outputs first disagree, so a failure names a line rather than printing two
// thousand of them.
func firstDifference(a, b string) string {
	la := strings.Split(a, "\n")
	lb := strings.Split(b, "\n")
	for i := range la {
		if i >= len(lb) || la[i] != lb[i] {
			lo := i - 2
			if lo < 0 {
				lo = 0
			}
			hi := i + 3
			if hi > len(la) {
				hi = len(la)
			}
			return strings.Join(la[lo:hi], "\n")
		}
	}
	if len(lb) > len(la) {
		return "(shorter by " + strings.Join(lb[len(la):], " / ") + ")"
	}
	return a
}
