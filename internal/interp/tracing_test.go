package interp_test

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/martin-k-m/twill/internal/interp"
)

// run executes f with tracing on or off and returns everything it printed.
func runTraced(t *testing.T, file string, tracing bool) string {
	t.Helper()
	var out strings.Builder
	ip := interp.New(func(s string) { out.WriteString(s); out.WriteByte('\n') })
	ip.SetTracing(tracing)
	if _, _, err := ip.RunFileMain(file, nil); err != nil {
		t.Fatalf("tracing=%v: %v", tracing, err)
	}
	return out.String()
}

func programs(t *testing.T) []string {
	t.Helper()
	var files []string
	for _, dir := range []string{"examples", filepath.Join("bench", "workloads")} {
		found, err := filepath.Glob(filepath.Join("..", "..", dir, "*.tw"))
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, found...)
	}
	if len(files) == 0 {
		t.Fatal("no programs found")
	}
	return files
}

// amplifiers are programs whose output is not reproducible across the two
// gradient paths, and the reason is not a defect. ir.Grad sums a multi-consumer
// cotangent in reverse node order where tensor.Backward sums it in DFS-visit
// order; docs/BENCHMARKS.md measured that at 113 of 1,606 gradients differing,
// worst case 3.74e-15. A few hundred optimiser steps turn that into the third
// significant figure. Both answers are correct and neither is the true one.
var amplifiers = map[string]bool{"signal_opt.tw": true}

// The tracer's whole claim is that it does not change an answer, so every
// program is run both ways and the output compared byte for byte. A compiled
// kernel that computes something subtly different fails here rather than being
// found later.
func TestTracingDoesNotChangeAnyOutput(t *testing.T) {
	for _, f := range programs(t) {
		if amplifiers[filepath.Base(f)] {
			continue
		}
		t.Run(filepath.Base(f), func(t *testing.T) {
			if got, want := runTraced(t, f, true), runTraced(t, f, false); got != want {
				t.Errorf("tracing changed the output of %s\n with tracing:\n%s\nwithout:\n%s",
					filepath.Base(f), got, want)
			}
		})
	}
}

// An amplifier still has to converge to the same place. This is the weaker claim
// that survives a different summation order, and it is worth asserting because
// the alternative to a loose bound here is no check at all on the one program
// that exercises the compiled backward pass hardest.
func TestAnAmplifierStillConvergesToTheSameAnswer(t *testing.T) {
	for name := range amplifiers {
		f := filepath.Join("..", "..", "examples", name)
		traced, plain := runTraced(t, f, true), runTraced(t, f, false)
		a, b := finalSharpe(t, traced), finalSharpe(t, plain)
		if rel := math.Abs(a-b) / math.Abs(b); rel > 0.01 {
			t.Errorf("%s: final Sharpe %v traced vs %v interpreted, %.2f%% apart",
				name, a, b, rel*100)
		}
	}
}

func finalSharpe(t *testing.T, out string) float64 {
	t.Helper()
	const prefix = "final strategy Sharpe: "
	for _, line := range strings.Split(out, "\n") {
		if rest, ok := strings.CutPrefix(line, prefix); ok {
			v, err := strconv.ParseFloat(strings.TrimSpace(rest), 64)
			if err != nil {
				t.Fatalf("parsing %q: %v", line, err)
			}
			return v
		}
	}
	t.Fatalf("no line starting %q in:\n%s", prefix, out)
	return 0
}

// A tracer that silently never traces would pass the test above. This fails if
// the corpus stops exercising it at all.
func TestTracingActuallyTracesSomething(t *testing.T) {
	traced := 0
	for _, f := range programs(t) {
		var out strings.Builder
		ip := interp.New(func(s string) { out.WriteString(s) })
		ip.SetTracing(true)
		if _, _, err := ip.RunFileMain(f, nil); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		if ip.TraceStats().Nodes > 0 {
			traced++
		}
	}
	if traced == 0 {
		t.Fatal("no program in the corpus produced a single traced node")
	}
	t.Logf("%d of %d programs produced traced nodes", traced, len(programs(t)))
}

func TestTracingSurvivesAProgramThatForcesConstantly(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "forcing.tw")
	src := `let acc = zeros(4)
for i in range(8) {
  acc = acc + ones(4)
  print(sum(acc))
}
print(sum(acc))
`
	if err := os.WriteFile(file, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, want := runTraced(t, file, true), runTraced(t, file, false); got != want {
		t.Errorf("forcing every iteration changed the answer\ngot:\n%s\nwant:\n%s", got, want)
	}
}
