package interp_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/martin-k-m/raster/internal/format"
	"github.com/martin-k-m/raster/internal/interp"
)

// TestFormattedExamplesMatch formats each deterministic example and checks that
// running the formatted source produces the same output as the original — so
// formatting is proven to preserve behavior, not just re-parse.
func TestFormattedExamplesMatch(t *testing.T) {
	files, _ := filepath.Glob(filepath.Join("..", "..", "examples", "*.ra"))
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		// Skip examples whose output isn't reproducible (any randomness, including
		// the random initializers from the nn library).
		s := string(src)
		if strings.Contains(s, "randn") || strings.Contains(s, "rand(") ||
			strings.Contains(s, "he_init") || strings.Contains(s, "xavier_init") {
			continue
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
