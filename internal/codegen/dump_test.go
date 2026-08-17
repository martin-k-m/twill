package codegen_test

import (
	"testing"

	"github.com/twill-lang/twill/internal/codegen"
	"github.com/twill-lang/twill/internal/ir"
)

// TestDumpSource is documentation that cannot go stale: it prints the C the
// backend emits for a small fused chain into a reduction, which is the shape
// the whole design is aimed at. Run it with -v to read the output.
func TestDumpSource(t *testing.T) {
	b := ir.NewBuilder()
	x := b.Param("x", []int{16})
	k := b.Param("k", []int{})
	b.Output(b.Mean(b.Unary(ir.OpRelu, b.Binary(ir.OpSub, b.Unary(ir.OpExp, x), k))))
	g, err := b.Finish()
	if err != nil {
		t.Fatal(err)
	}
	src, lay, err := codegen.Emit(ir.Fuse(g))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("arena: %d f64\n%s", lay.Total, src)
}
