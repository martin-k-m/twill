package interp_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/martin-k-m/raster/internal/interp"
)

func TestRecordLiteralAndField(t *testing.T) {
	if got := scalar(t, "let p = { a: 3.0, b: 4.0 }\np.a + p.b"); got != 7 {
		t.Errorf("got %v, want 7", got)
	}
	// Nested records.
	if got := scalar(t, "let r = { inner: { x: 5.0 } }\nr.inner.x"); got != 5 {
		t.Errorf("nested got %v, want 5", got)
	}
}

func TestRecordFieldTensor(t *testing.T) {
	if got := scalar(t, "let m = { w: [1.0, 2.0, 3.0] }\nsum(m.w)"); got != 6 {
		t.Errorf("got %v, want 6", got)
	}
}

func TestGradThroughRecord(t *testing.T) {
	// loss(m) = sum(m.w) + m.b ; d/dw = ones, d/db = 1.
	src := "fn loss(m) = sum(m.w) + m.b\ngrad(loss)({ w: [1.0, 2.0], b: 0.5 }).w[1]"
	if got := scalar(t, src); got != 1 {
		t.Errorf("d/dw got %v, want 1", got)
	}
	src2 := "fn loss(m) = sum(m.w) + m.b\ngrad(loss)({ w: [1.0, 2.0], b: 0.5 }).b"
	if got := scalar(t, src2); got != 1 {
		t.Errorf("d/db got %v, want 1", got)
	}
}

func TestRecordFormatPreservesOrder(t *testing.T) {
	_, out := run(t, `print({ z: 1.0, a: 2.0 })`)
	if len(out) != 1 || out[0] != "{z: 1, a: 2}" {
		t.Errorf("got %q", out)
	}
}

func TestNamespacedImport(t *testing.T) {
	dir := t.TempDir()
	lib := filepath.Join(dir, "lib.ra")
	if err := os.WriteFile(lib, []byte("fn triple(x) = x * 3.0\nlet k = 7.0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	main := filepath.Join(dir, "main.ra")
	body := "import \"lib.ra\" as lib\nprint(lib.triple(4.0))\nprint(lib.k)\n"
	if err := os.WriteFile(main, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	var out []string
	ip := interp.New(func(s string) { out = append(out, s) })
	if err := ip.RunFile(main); err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0] != "12" || out[1] != "7" {
		t.Errorf("namespaced import output = %q", out)
	}
}
