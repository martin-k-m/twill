package interp_test

import (
	"testing"

	"github.com/martin-k-m/raster/internal/interp"
)

func TestMapLeavesRecord(t *testing.T) {
	if got := scalar(t, "map_leaves(fn(p) = p * 2.0, { a: [1.0], b: [2.0, 3.0] }).b[1]"); got != 6 {
		t.Errorf("got %v, want 6", got)
	}
}

func TestZipLeavesRecord(t *testing.T) {
	src := "zip_leaves(fn(z) = z[0] + z[1], [{ a: [1.0, 2.0] }, { a: [10.0, 20.0] }]).a[1]"
	if got := scalar(t, src); got != 22 {
		t.Errorf("got %v, want 22", got)
	}
}

func TestZipLeavesList(t *testing.T) {
	// The same helper works on list-structured trees (a list of tensors).
	src := "let a = list([1.0], [2.0])\nlet b = list([10.0], [20.0])\nzip_leaves(fn(z) = z[0] + z[1], [a, b])[1][0]"
	if got := scalar(t, src); got != 22 {
		t.Errorf("got %v, want 22", got)
	}
}

func TestZipLeavesStructureMismatch(t *testing.T) {
	ip := interp.New(func(string) {})
	if _, err := ip.Run("zip_leaves(fn(z) = z[0], [{ a: [1.0] }, [1.0]])"); err == nil {
		t.Fatal("expected a structure-mismatch error")
	}
}

func TestRecordModelTrainsWithLibraryAdam(t *testing.T) {
	// Train a record-structured model with the library's generic Adam, which
	// walks the record via map_leaves/zip_leaves. Loss should converge to ~0.
	src := `
		import "../../std/optim.ra"
		fn loss(model) = sum(model.w * model.w) + model.b * model.b
		let model = { w: [3.0, -4.0], b: 2.0 }
		let m = zeros_like(model)
		let v = zeros_like(model)
		for step in range(400) {
			let g = grad(loss)(model)
			let out = adam_step(model, g, m, v, step + 1, 0.1, 0.9, 0.999, 0.00000001)
			model = out[0]
			m = out[1]
			v = out[2]
		}
		loss(model)`
	if got := scalar(t, src); got > 0.001 {
		t.Errorf("loss did not converge: %v", got)
	}
}
