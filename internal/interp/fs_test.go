package interp_test

import (
	"path/filepath"
	"strings"
	"testing"
)

// The filesystem primitives added in 1.6 (docs/needs.md NEEDS-91, NEEDS-92).
// Everything that can fail for a reason a caller may handle returns a Res; the
// two predicates return a Bool, because "it is not there" is an answer.
func TestFilesystemPrimitives(t *testing.T) {
	dir := t.TempDir()
	out := runFile(t, dir, `mode systems
fn main() {
  let base: Str = match temp_dir("twilltest") { Ok(p) => p, Err(e) => abort(e) }
  print(path_exists(base), path_is_dir(base))
  let sub: Str = path_join(base, "a", "b")
  match mkdir_all(sub) { Ok(_) => print("made"), Err(e) => print(e) }
  print(path_is_dir(sub))
  let f: Str = path_join(sub, "note.txt")
  match write_file(f, "hello") { Ok(_) => print("wrote"), Err(e) => print(e) }
  print(path_exists(f), path_is_dir(f), file_size(f))
  match read_file(f) { Ok(s) => print(s), Err(e) => print(e) }
  let g: Str = path_join(sub, "moved.txt")
  match rename(f, g) { Ok(_) => print("moved"), Err(e) => print(e) }
  print(path_exists(f), path_exists(g))
  match remove_file(sub) { Ok(_) => print("BAD: removed a directory"), Err(_) => print("refused") }
  match remove_file(g) { Ok(_) => print("removed"), Err(e) => print(e) }
  match remove_dir(base) { Ok(_) => print("cleaned"), Err(e) => print(e) }
  print(path_exists(base))
  # remove_all takes either, and succeeds on a path that is already gone.
  match remove_all(base) { Ok(_) => print("gone"), Err(e) => print(e) }
}
`)
	expectLines(t, out,
		"true true", "made", "true", "wrote", "true false 5", "hello",
		"moved", "false true", "refused", "removed", "cleaned", "false", "gone")
}

// A failure is a value, not a crash: every one of these reports an Err naming
// what went wrong.
func TestFilesystemFailuresAreValues(t *testing.T) {
	dir := t.TempDir()
	missing := strings.ReplaceAll(filepath.Join(dir, "nope", "gone.txt"), `\`, `/`)
	out := runFile(t, dir, `mode systems
fn describe(r: Res[Unit, Str]) -> Str {
  match r { Ok(_) => "ok", Err(_) => "err" }
}
fn main() {
  print(describe(remove_file("`+missing+`")))
  print(describe(remove_dir("`+missing+`")))
  print(describe(rename("`+missing+`", "`+missing+`2")))
  print(describe(remove_all("`+missing+`")))
  print(path_exists("`+missing+`"), path_is_dir("`+missing+`"))
}
`)
	expectLines(t, out, "err", "err", "err", "ok", "false false")
}

// mtime and mono_ns: when a file last changed, and a clock that only goes
// forward. mono_ns's zero point is arbitrary, so only a difference is asserted.
func TestTimestamps(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
fn main() {
  let a: I64 = mono_ns()
  let b: I64 = mono_ns()
  print(b >= a)
  print(mtime("no-such-file-anywhere") == -1)
  let d: Str = match temp_dir("twillmt") { Ok(p) => p, Err(e) => abort(e) }
  let f: Str = path_join(d, "x.txt")
  match write_file(f, "hi") { Ok(_) => unit, Err(e) => print(e) }
  print(mtime(f) > 1700000000)
  match remove_all(d) { Ok(_) => print("gone"), Err(e) => print(e) }
}
`)
	expectLines(t, out, "true", "true", "true", "gone")
}

// The path operations are string handling: they never touch the filesystem,
// they emit a forward slash on every platform, and they read a backslash as a
// separator so a path handed over by the operating system still splits.
func TestPathOperations(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
fn main() {
  print(path_join("a", "b", "c.txt"))
  print(path_join("a/", "/b"))
  print(path_join("a", "", "b"))
  print(path_base("a/b/c.txt"), path_dir("a/b/c.txt"))
  print(path_ext("a/b/c.txt"), path_stem("a/b/c.txt"))
  print(path_ext("a/b/c"), path_stem("a/b/c"))
  print(path_ext("a/.hidden"), path_stem("a/.hidden"))
  print(path_normalize("a/x/../b/./c"))
  print(path_normalize("a/../.."))
  print(path_base("a\\b\\c.txt"), path_dir("a\\b\\c.txt"))
  print(path_dir("c.txt"), path_base("a/b/"))
  print(path_is_abs("/x/y"), path_is_abs("x/y"))
}
`)
	expectLines(t, out,
		"a/b/c.txt", "a/b", "a/b",
		"c.txt a/b", ".txt c", " c", " .hidden",
		"a/b/c", "..",
		"c.txt a/b", ". b",
		"true false")
}

// cwd is a Res, so a caller reaches its answer the same way as any other
// failable one.
func TestCwd(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
fn main() {
  match cwd() { Ok(p) => print(len(p) > 0), Err(e) => print(e) }
}
`)
	expectLines(t, out, "true")
}
