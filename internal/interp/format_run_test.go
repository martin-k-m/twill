package interp_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/martin-k-m/twill/internal/format"
	"github.com/martin-k-m/twill/internal/interp"
)

// TestFormattedExamplesMatch formats each example and checks that running the
// formatted source produces the same output as the original — so formatting is
// proven to preserve behavior, not just re-parse. Randomness is deterministic
// by default, so even the stochastic examples reproduce.
func TestFormattedExamplesMatch(t *testing.T) {
	files, _ := filepath.Glob(filepath.Join("..", "..", "examples", "*.tw"))
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		f := f
		t.Run(filepath.Base(f), func(t *testing.T) {
			formatted, err := format.Source(string(src))
			if err != nil {
				t.Fatalf("format: %v", err)
			}
			orig := runCapture(t, f, string(src))
			// Write the formatted source next to the original so relative
			// imports still resolve, then run it.
			dir := filepath.Dir(f)
			tmp := filepath.Join(dir, ".fmt_"+filepath.Base(f))
			if err := os.WriteFile(tmp, []byte(formatted), 0o644); err != nil {
				t.Fatal(err)
			}
			defer os.Remove(tmp)
			got := runCapture(t, tmp, formatted)
			if strings.Join(orig, "\n") != strings.Join(got, "\n") {
				t.Errorf("formatted output differs for %s:\n--- original ---\n%s\n--- formatted ---\n%s",
					f, strings.Join(orig, "\n"), strings.Join(got, "\n"))
			}
		})
	}
}

func runCapture(t *testing.T, path, _ string) []string {
	t.Helper()
	var out []string
	ip := interp.New(func(s string) { out = append(out, s) })
	if err := ip.RunFile(path); err != nil {
		t.Fatalf("run %s: %v", path, err)
	}
	return out
}
