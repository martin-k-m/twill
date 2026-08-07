package interp_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/martin-k-m/twill/internal/interp"
)

func TestReadCSV(t *testing.T) {
	dir := t.TempDir()
	csv := "1, 2, 3\n4 5 6\n# a comment\n7,8,9\n"
	if err := os.WriteFile(filepath.Join(dir, "data.csv"), []byte(csv), 0o644); err != nil {
		t.Fatal(err)
	}
	prog := `
		let d = read_csv("data.csv")
		print(shape(d)[0], shape(d)[1])
		print(d[1][2])
		print(sum(d))
	`
	main := filepath.Join(dir, "main.tw")
	if err := os.WriteFile(main, []byte(prog), 0o644); err != nil {
		t.Fatal(err)
	}
	var out []string
	ip := interp.New(func(s string) { out = append(out, s) })
	if err := ip.RunFile(main); err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 || out[0] != "3 3" || out[1] != "6" || out[2] != "45" {
		t.Errorf("read_csv output = %q", out)
	}
}
