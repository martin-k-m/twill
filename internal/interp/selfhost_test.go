package interp_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/martin-k-m/twill/internal/interp"
	"github.com/martin-k-m/twill/internal/value"
)

// The self-hosted compiler is twill written in twill (src/main.tw and the
// modules it imports). Running it on the Go bootstrap and asking it to check a
// file exercises the whole front end -- lexer, parser and checker, all in twill
// -- end to end. Its exit code is the milestone under guard: 0 for a clean file,
// 1 for one with a shape error, matching what the Go `check` command returns.
func runSelfHostedCheck(t *testing.T, source string) int {
	t.Helper()
	dir := t.TempDir()
	target := filepath.Join(dir, "input.tw")
	if err := os.WriteFile(target, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	ip := interp.New(func(string) {})
	result, ranMain, err := ip.RunFileMain(filepath.Join("..", "..", "src", "main.tw"),
		[]string{"twill", "check", target})
	if err != nil {
		t.Fatalf("self-hosted CLI errored: %v", err)
	}
	if !ranMain {
		t.Fatal("self-hosted main did not run")
	}
	n, ok := value.AsNumber(result)
	if !ok {
		t.Fatalf("self-hosted main returned a non-number: %v", result)
	}
	return int(n)
}

func TestSelfHostedCheckCleanFile(t *testing.T) {
	if code := runSelfHostedCheck(t, "mode systems\nfn add(a: I64, b: I64) -> I64 = a + b\n"); code != 0 {
		t.Fatalf("check of a clean file exited %d, want 0", code)
	}
}

func TestSelfHostedCheckShapeError(t *testing.T) {
	src := "let a = [1.0, 2.0]\nlet b = [1.0, 2.0, 3.0]\nlet c = a + b\n"
	if code := runSelfHostedCheck(t, src); code != 1 {
		t.Fatalf("check of a mismatched file exited %d, want 1", code)
	}
}
