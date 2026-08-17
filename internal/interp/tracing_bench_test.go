package interp_test

import (
	"path/filepath"
	"testing"

	"github.com/twill-lang/twill/internal/interp"
)

// The pair to run when changing the tracer. In-process, so it measures the
// tracer rather than process startup and DLL loading; docs/CODEGEN.md section 11
// has the end-to-end numbers, which are worse and are the ones to quote.
func benchProgram(b *testing.B, name string, tracing bool) {
	f := filepath.Join("..", "..", "examples", name)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ip := interp.New(func(string) {})
		ip.SetTracing(tracing)
		if _, _, err := ip.RunFileMain(f, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLinregTraced(b *testing.B)   { benchProgram(b, "linreg.tw", true) }
func BenchmarkLinregInterp(b *testing.B)   { benchProgram(b, "linreg.tw", false) }
func BenchmarkMLPTraced(b *testing.B)      { benchProgram(b, "mlp.tw", true) }
func BenchmarkMLPInterp(b *testing.B)      { benchProgram(b, "mlp.tw", false) }
func BenchmarkMCOptionTraced(b *testing.B) { benchProgram(b, "montecarlo_option.tw", true) }
func BenchmarkMCOptionInterp(b *testing.B) { benchProgram(b, "montecarlo_option.tw", false) }
