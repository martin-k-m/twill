package interp_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/twill-lang/twill/internal/checker"
	"github.com/twill-lang/twill/internal/interp"
	"github.com/twill-lang/twill/internal/parser"
)

// TestExamplesRunClean shape-checks and runs every example program in-process,
// so the suite covers them without depending on the built binary.
func TestExamplesRunClean(t *testing.T) {
	dir := filepath.Join("..", "..", "examples")
	files, err := filepath.Glob(filepath.Join(dir, "*.tw"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no example files found in %s (err=%v)", dir, err)
	}
	for _, f := range files {
		f := f
		t.Run(filepath.Base(f), func(t *testing.T) {
			src, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			prog, perr := parser.Parse(string(src))
			if perr != nil {
				t.Fatalf("parse: %v", perr)
			}
			if diags := checker.Check(prog); len(diags) != 0 {
				t.Fatalf("unexpected shape diagnostics: %v", diags)
			}
			ip := interp.New(func(string) {}) // discard output
			if _, _, err := ip.RunFileMain(f, nil); err != nil {
				t.Fatalf("run: %v", err)
			}
		})
	}
}
