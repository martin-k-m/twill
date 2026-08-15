package ir_test

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/martin-k-m/twill/internal/ir"
	"github.com/martin-k-m/twill/internal/parser"
)

// How much of twill is inside the compilable subset, measured over the corpus
// rather than estimated.
//
// The number this reports is a ceiling and the test says so in its output,
// because there is no AST-to-IR front end: it classifies forms against the
// opcode set, it does not compile anything. What it is good for is keeping the
// claim honest in both directions. A doc that said "the compiler handles twill"
// would be contradicted by the per-file column, and a doc that said the subset
// was tiny would be contradicted by the aggregate.
func TestCompilableSubsetOverTheCorpus(t *testing.T) {
	roots := []string{"../../examples", "../../bench/workloads", "../../std"}
	type row struct {
		path  string
		cov   ir.Coverage
		clean bool
	}
	var rows []row
	total := ir.Coverage{Reasons: map[string]int{}}
	for _, root := range roots {
		err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() || filepath.Ext(path) != ".tw" {
				return nil
			}
			src, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			prog, perr := parser.Parse(string(src))
			if perr != nil {
				t.Logf("skipping %s: %v", path, perr)
				return nil
			}
			c := ir.CoverProgram(prog)
			total.Add(c)
			rows = append(rows, row{filepath.ToSlash(path), c, c.Clean})
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(rows) == 0 {
		t.Fatal("no .tw files found")
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].cov.Fraction() > rows[j].cov.Fraction() })

	// Per directory, because the aggregate mixes three different kinds of file.
	// bench/workloads is numeric kernels, examples is programs with printing and
	// IO around the numerics, and std is a library with strings, records, lists
	// and control flow in it.
	byRoot := map[string]*ir.Coverage{}
	for _, r := range rows {
		key := filepath.ToSlash(filepath.Dir(r.path))
		for _, root := range roots {
			if strings.HasPrefix(r.path, filepath.ToSlash(root)+"/") {
				key = filepath.ToSlash(root)
			}
		}
		if byRoot[key] == nil {
			byRoot[key] = &ir.Coverage{Reasons: map[string]int{}}
		}
		byRoot[key].Add(r.cov)
	}
	keys := make([]string, 0, len(byRoot))
	for k := range byRoot {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	clean := 0
	for _, r := range rows {
		if r.clean {
			clean++
		}
	}
	t.Logf("this is a static classification against the opcode set, not a compilation:")
	t.Logf("%d files, %d entirely inside the compilable subset (%.1f%%)",
		len(rows), clean, 100*float64(clean)/float64(len(rows)))
	t.Logf("%d of %d classified nodes inside the subset (%.1f%%)",
		total.In, total.In+total.Out, 100*total.Fraction())
	for _, k := range keys {
		c := byRoot[k]
		t.Logf("  %-24s %d/%d nodes inside (%.1f%%)", k, c.In, c.In+c.Out, 100*c.Fraction())
	}
	t.Logf("what falls outside, most frequent first:")
	for _, r := range total.TopReasons() {
		t.Logf("  %s", r)
	}
	t.Logf("per file:")
	for _, r := range rows {
		mark := " "
		if r.clean {
			mark = "*"
		}
		t.Logf("  %s %-44s %5.1f%%", mark, r.path, 100*r.cov.Fraction())
	}
}
