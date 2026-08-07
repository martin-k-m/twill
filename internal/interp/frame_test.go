package interp_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/martin-k-m/twill/internal/interp"
)

func runFileCapture(t *testing.T, dir, name, src string) []string {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	var out []string
	ip := interp.New(func(s string) { out = append(out, s) })
	if err := ip.RunFile(filepath.Join(dir, name)); err != nil {
		t.Fatalf("run: %v", err)
	}
	return out
}

func TestReadFrame(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "d.csv"), []byte("a,b\n1,2\n3,4\n5,6\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out := runFileCapture(t, dir, "main.tw",
		`let df = read_frame("d.csv")`+"\n"+`print(len(df.a), sum(df.a), sum(df.b))`)
	if len(out) != 1 || out[0] != "3 9 12" {
		t.Errorf("read_frame output = %q, want %q", out, "3 9 12")
	}
}

func TestWriteReadFrameRoundTrip(t *testing.T) {
	dir := t.TempDir()
	src := `
		let df = { x: [1.0, 2.0, 3.0], y: [10.0, 20.0, 30.0] }
		write_frame(df, "out.csv")
		let df2 = read_frame("out.csv")
		print(columns(df2), sum(df2.x), sum(df2.y))`
	out := runFileCapture(t, dir, "main.tw", src)
	if len(out) != 1 || out[0] != "[x, y] 6 60" {
		t.Errorf("round-trip output = %q, want %q", out, "[x, y] 6 60")
	}
	// The written file should have a header and the rows.
	written, _ := os.ReadFile(filepath.Join(dir, "out.csv"))
	if !strings.HasPrefix(string(written), "x,y\n") {
		t.Errorf("written CSV missing header: %q", string(written))
	}
}

func TestColumnFieldOps(t *testing.T) {
	if got := scalar(t, `field({ a: 3.0, b: 4.0 }, "b")`); got != 4 {
		t.Errorf("field got %v", got)
	}
	if got := scalar(t, `with_field({ a: 1.0 }, "c", 9.0).c`); got != 9 {
		t.Errorf("with_field got %v", got)
	}
	if got := scalar(t, `len(columns({ a: 1.0, b: 2.0, c: 3.0 }))`); got != 3 {
		t.Errorf("columns len got %v", got)
	}
	// with_field replaces an existing field without duplicating it.
	if got := scalar(t, `len(columns(with_field({ a: 1.0 }, "a", 2.0)))`); got != 1 {
		t.Errorf("with_field replace got %v cols, want 1", got)
	}
}
