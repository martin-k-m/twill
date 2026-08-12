package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseSummaryLine(t *testing.T) {
	cases := []struct {
		line      string
		p, f      int
		wantMatch bool
	}{
		{"nn_test passed 52 failed 0", 52, 0, true},
		{"random passed 17 failed 18", 17, 18, true},
		{"  frame_test passed 54 failed 0  ", 54, 0, true},
		{"just some output", 0, 0, false},
		{"passed but no number failed 3", 0, 0, false},
	}
	for _, c := range cases {
		p, f, ok := parseSummaryLine(strings.TrimSpace(c.line))
		if ok != c.wantMatch || (ok && (p != c.p || f != c.f)) {
			t.Errorf("parseSummaryLine(%q) = (%d,%d,%v), want (%d,%d,%v)",
				c.line, p, f, ok, c.p, c.f, c.wantMatch)
		}
	}
}

// writeTemp writes a .tw file into a temp dir and returns its path.
func writeTemp(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "x_test.tw")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRunOneTestFilePasses(t *testing.T) {
	// A file whose harness summary reports zero failures passes.
	res := runOneTestFile(writeTemp(t, "print(\"x passed 3 failed 0\")\nprint(\"OK\")\n"))
	if !res.ok {
		t.Fatalf("expected pass, got fail: %q %q", res.output, res.errMsg)
	}
	if !res.counted || res.checksP != 3 || res.checksF != 0 {
		t.Fatalf("expected counts (3,0), got (%d,%d) counted=%v", res.checksP, res.checksF, res.counted)
	}
}

func TestRunOneTestFileFailsOnMarker(t *testing.T) {
	// A FAILED marker, or a nonzero failed count, is a failing file even though
	// the program itself ran without error.
	res := runOneTestFile(writeTemp(t, "print(\"x passed 2 failed 1\")\nprint(\"FAILED\")\n"))
	if res.ok {
		t.Fatalf("expected fail, got pass")
	}
	if res.checksF != 1 {
		t.Fatalf("expected 1 failed check, got %d", res.checksF)
	}
}

func TestRunOneTestFileFailsOnRuntimeError(t *testing.T) {
	// An out-of-range index aborts; a file that cannot finish is a failing file.
	res := runOneTestFile(writeTemp(t, "let xs = [1.0]\nprint(xs[9])\n"))
	if res.ok {
		t.Fatalf("expected fail on runtime error, got pass")
	}
	if res.errMsg == "" {
		t.Fatalf("expected an error message")
	}
}

func TestRunOneTestFileFailsOnShapeError(t *testing.T) {
	res := runOneTestFile(writeTemp(t, "let a = [1.0, 2.0]\nlet b = [1.0, 2.0, 3.0]\nlet c = a + b\n"))
	if res.ok {
		t.Fatalf("expected fail on shape error, got pass")
	}
	if !strings.Contains(res.errMsg, "shape error") {
		t.Fatalf("expected a shape-error message, got %q", res.errMsg)
	}
}

func TestDiscoverFindsTestFilesRecursively(t *testing.T) {
	dir := t.TempDir()
	sub := filepath.Join(dir, "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"a_test.tw", "sub/b_test.tw", "notatest.tw", "c.txt"} {
		if err := os.WriteFile(filepath.Join(dir, f), []byte("print(1)\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	files, err := discoverTestFiles([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 {
		t.Fatalf("expected 2 *_test.tw files, got %d: %v", len(files), files)
	}
}

// The numeric-mode standard-library suites are the regression guard: they run on
// the bootstrap as-is and must all pass. The systems-mode suites (io, json,
// random, ...) are excluded here because some are written ahead of the language
// -- random needs true 64-bit integers the f64 bootstrap does not have -- and
// their status is tracked separately, not as a regression. See std/tests/README.
func TestNumericStdSuitesPass(t *testing.T) {
	root := filepath.Join("..", "..", "std", "tests")
	files, err := discoverTestFiles([]string{root})
	if err != nil {
		t.Fatal(err)
	}
	ran := 0
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if strings.HasPrefix(strings.TrimSpace(firstLine(string(src))), "mode systems") {
			continue // systems-mode suite: best-effort, guarded elsewhere
		}
		ran++
		res := runOneTestFile(f)
		if !res.ok {
			t.Errorf("%s failed:\n%s\n%s", filepath.Base(f), res.output, res.errMsg)
		}
	}
	if ran == 0 {
		t.Fatal("found no numeric std test suites to run")
	}
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
