package interp_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/martin-k-m/raster/internal/interp"
	"github.com/martin-k-m/raster/internal/value"
)

// writeModule writes src to dir/name.ra and returns the path.
func writeModule(t *testing.T, dir, name, src string) string {
	t.Helper()
	path := filepath.Join(dir, name+".ra")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// runFile runs a program written to a temp file, so imports resolve relative to
// it, and returns its printed output.
func runFile(t *testing.T, dir, src string) []string {
	t.Helper()
	main := writeModule(t, dir, "main", src)
	var out []string
	ip := interp.New(func(s string) { out = append(out, s) })
	if err := ip.RunFile(main); err != nil {
		t.Fatalf("run error: %v\nsource:\n%s", err, src)
	}
	return out
}

// A namespace record's fields are in the module's declaration order. They used
// to come out in Go map order, so the same program printed a different record
// on different runs.
func TestNamespacedImportFieldOrderIsDeclarationOrder(t *testing.T) {
	dir := t.TempDir()
	writeModule(t, dir, "lib", "let zeta = 1.0\nfn mid(x) = x\nlet alpha = 3.0\nlet beta = 4.0\n")
	const want = "[zeta, mid, alpha, beta]"
	// Map iteration order varies per range, not per process, so repeat enough
	// that an unordered snapshot could not stay lucky.
	for i := 0; i < 50; i++ {
		out := runFile(t, dir, "import \"lib.ra\" as lib\nprint(columns(lib))\n")
		if len(out) != 1 || out[0] != want {
			t.Fatalf("run %d: field order = %q, want %q", i, out, want)
		}
	}
}

// Definitions a module picks up from its own plain imports are part of its
// namespace, and keep their place in the order too.
func TestNamespacedImportOrderIncludesNestedImports(t *testing.T) {
	dir := t.TempDir()
	writeModule(t, dir, "base", "let first = 1.0\nlet second = 2.0\n")
	writeModule(t, dir, "lib", "import \"base.ra\"\nlet third = 3.0\n")
	const want = "[first, second, third]"
	for i := 0; i < 20; i++ {
		out := runFile(t, dir, "import \"lib.ra\" as lib\nprint(columns(lib))\n")
		if len(out) != 1 || out[0] != want {
			t.Fatalf("run %d: field order = %q, want %q", i, out, want)
		}
	}
}

// The standard library is read out of the binary, so `import "std/..."` works
// from any directory rather than depending on where std/ sits on disk.
func TestStdImportIsIndependentOfWorkingDirectory(t *testing.T) {
	dir := t.TempDir()
	out := runFile(t, dir, "import \"std/nn\" as nn\nprint(nn.mse([1.0, 2.0], [1.0, 4.0]))\n")
	if len(out) != 1 || out[0] != "2" {
		t.Fatalf("std import output = %q", out)
	}
}

// std/nn pulls in std/optim, so its namespace carries the optimizers too, in a
// stable order.
func TestStdNamespaceOrderIsStable(t *testing.T) {
	dir := t.TempDir()
	first := runFile(t, dir, "import \"std/nn\" as nn\nprint(columns(nn))\n")
	if len(first) != 1 || !strings.HasPrefix(first[0], "[zeros_like, sgd_step,") {
		t.Fatalf("std/nn field order = %q", first)
	}
	for i := 0; i < 20; i++ {
		if got := runFile(t, dir, "import \"std/nn\" as nn\nprint(columns(nn))\n"); got[0] != first[0] {
			t.Fatalf("run %d: %q, want %q", i, got[0], first[0])
		}
	}
}

// The old spelling carried a file extension. It is gone, and says so.
func TestStdImportRejectsFileExtension(t *testing.T) {
	ip := interp.New(func(string) {})
	_, err := ip.Run(`import "std/nn.ra"`)
	if err == nil {
		t.Fatal("expected an error for the old std/nn.ra spelling")
	}
	if !strings.Contains(err.Error(), `write "std/nn"`) {
		t.Errorf("error should point at the new spelling, got %q", err)
	}
}

func TestUnknownStdModuleListsTheLibrary(t *testing.T) {
	ip := interp.New(func(string) {})
	_, err := ip.Run(`import "std/nope"`)
	if err == nil {
		t.Fatal("expected an error for an unknown std module")
	}
	if !strings.Contains(err.Error(), "backtest") || !strings.Contains(err.Error(), "optim") {
		t.Errorf("error should list the available modules, got %q", err)
	}
}

// std/ is reserved: a directory of the same name next to the program does not
// shadow the library.
func TestStdPrefixIsNotShadowedByALocalDirectory(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "std"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeModule(t, dir, filepath.Join("std", "nn"), "fn mse(a, b) = 999.0\n")
	out := runFile(t, dir, "import \"std/nn\" as nn\nprint(nn.mse([1.0], [1.0]))\n")
	if len(out) != 1 || out[0] != "0" {
		t.Fatalf("local std/ shadowed the embedded library: %q", out)
	}
}

// RASTER_STD is the escape hatch: it replaces the embedded library wholesale.
func TestStdOverrideDirectory(t *testing.T) {
	over := t.TempDir()
	writeModule(t, over, "nn", "fn mse(a, b) = 42.0\n")
	t.Setenv("RASTER_STD", over)

	dir := t.TempDir()
	out := runFile(t, dir, "import \"std/nn\" as nn\nprint(nn.mse([1.0], [1.0]))\n")
	if len(out) != 1 || out[0] != "42" {
		t.Fatalf("override not used: %q", out)
	}

	ip := interp.New(func(string) {})
	if _, err := ip.Run(`import "std/optim"`); err == nil {
		t.Fatal("expected a missing-module error from the override directory")
	} else if !strings.Contains(err.Error(), "RASTER_STD") {
		t.Errorf("error should name the override, got %q", err)
	}
}

// A std module has no directory of its own, so it cannot reach the filesystem
// through a relative import. This also stops an override directory from
// pulling in code outside itself.
func TestStdModuleCannotImportAFile(t *testing.T) {
	over := t.TempDir()
	writeModule(t, over, "secret", "let leaked = 1.0\n")
	writeModule(t, over, "nn", "import \"secret.ra\"\n")
	t.Setenv("RASTER_STD", over)

	ip := interp.New(func(string) {})
	_, err := ip.Run(`import "std/nn"`)
	if err == nil {
		t.Fatal("expected a relative import inside std to be rejected")
	}
	if !strings.Contains(err.Error(), "may only import other std modules") {
		t.Errorf("unexpected error: %q", err)
	}
}

// A relative path that merely mentions std is still an ordinary file import.
func TestRelativePathToStdFileStillWorks(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "std"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeModule(t, dir, filepath.Join("std", "local"), "fn answer() = 7.0\n")
	out := runFile(t, dir, "import \"./std/local.ra\"\nprint(answer())\n")
	if len(out) != 1 || out[0] != "7" {
		t.Fatalf("relative import of a std-named directory = %q", out)
	}
}

// An imported module still evaluates once, whichever spelling reaches it.
func TestStdPlainImportLoadsOnce(t *testing.T) {
	ip := interp.New(func(string) {})
	v, err := ip.Run("import \"std/optim\"\nimport \"std/optim\"\nzeros_like([1.0, 2.0])")
	if err != nil {
		t.Fatal(err)
	}
	if got := value.Format(v); got != "tensor([0, 0], shape=[2])" {
		t.Errorf("got %s", got)
	}
}

// A plain import at the top level used to hollow out any namespace that
// imported the same module. The load-once set was global, so the nested plain
// import inside the namespaced module was skipped as already loaded and its
// names never reached the module scope: after `import "std/optim"`, the
// namespace from `import "std/nn" as nn` came back without any of optim's
// names. What a namespace contains must not depend on what was imported before
// it.
func TestAPlainImportDoesNotHollowOutALaterNamespace(t *testing.T) {
	dir := t.TempDir()
	writeModule(t, dir, "base", "let first = 1.0\nlet second = 2.0\n")
	writeModule(t, dir, "lib", "import \"base.ra\"\nlet third = 3.0\n")

	alone := runFile(t, dir, "import \"lib.ra\" as lib\nprint(columns(lib))\n")
	after := runFile(t, dir,
		"import \"base.ra\"\nimport \"lib.ra\" as lib\nprint(columns(lib))\n")

	if len(alone) != 1 || len(after) != 1 {
		t.Fatalf("expected one line from each run, got %q and %q", alone, after)
	}
	if alone[0] != after[0] {
		t.Fatalf("importing base first changed the namespace:\n  alone = %s\n  after = %s",
			alone[0], after[0])
	}
	if !strings.Contains(after[0], "first") {
		t.Fatalf("namespace lost the nested module's names: %s", after[0])
	}
}

// The same shape against the standard library, which is where it was found.
func TestAPlainStdImportDoesNotHollowOutALaterNamespace(t *testing.T) {
	dir := t.TempDir()
	alone := runFile(t, dir, "import \"std/nn\" as nn\nprint(columns(nn))\n")
	after := runFile(t, dir, "import \"std/optim\"\nimport \"std/nn\" as nn\nprint(columns(nn))\n")

	if len(alone) != 1 || len(after) != 1 {
		t.Fatalf("expected one line from each run, got %q and %q", alone, after)
	}
	if alone[0] != after[0] {
		t.Fatalf("importing optim first changed the nn namespace:\n  alone = %s\n  after = %s",
			alone[0], after[0])
	}
}

// The fresh load-once set per module scope must not lose the cycle guard: two
// files that plain-import each other, reached through a namespace, have to
// terminate rather than recurse until the stack goes.
func TestAPlainImportCycleInsideANamespaceTerminates(t *testing.T) {
	dir := t.TempDir()
	writeModule(t, dir, "x", "import \"y.ra\"\nlet a = 1.0\n")
	writeModule(t, dir, "y", "import \"x.ra\"\nlet b = 2.0\n")

	out := runFile(t, dir, "import \"x.ra\" as m\nprint(columns(m))\n")
	if len(out) != 1 {
		t.Fatalf("expected one line, got %q", out)
	}
	for _, name := range []string{"a", "b"} {
		if !strings.Contains(out[0], name) {
			t.Fatalf("cycle lost %q: %s", name, out[0])
		}
	}
}

// Two namespaces over the same module are separate bindings, so the second must
// not come back empty because the first already loaded it.
func TestTwoNamespacesOverOneModuleAreIndependent(t *testing.T) {
	dir := t.TempDir()
	writeModule(t, dir, "base", "let first = 1.0\n")
	writeModule(t, dir, "lib", "import \"base.ra\"\nlet third = 3.0\n")

	out := runFile(t, dir,
		"import \"lib.ra\" as p\nimport \"lib.ra\" as q\nprint(columns(p))\nprint(columns(q))\n")
	if len(out) != 2 {
		t.Fatalf("expected two lines, got %q", out)
	}
	if out[0] != out[1] {
		t.Fatalf("the two namespaces differ:\n  p = %s\n  q = %s", out[0], out[1])
	}
}
