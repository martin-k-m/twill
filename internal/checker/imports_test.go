package checker_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/twill-lang/twill/internal/checker"
	"github.com/twill-lang/twill/internal/parser"
)

// A match on an enum declared in another module could not be checked at all:
// the checker reads one file, so it did not know the enum's other cases and
// said nothing. Matching on an imported enum is not an edge case, it is how the
// ecosystem is written -- warp's pipeline kinds, weft's marks, spool's version
// constraints are declared in one module and matched in another -- so the check
// did nothing in exactly the place it was most wanted.

func checkFileIn(t *testing.T, dir, name, src string) []checker.Diagnostic {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	prog, err := parser.Parse(src)
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return checker.CheckFile(prog, path)
}

func writeFile(t *testing.T, dir, name, src string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
}

const kindModule = `mode systems
enum Kind { Map, Filter, Batch }
struct S { kind: Kind, n: I64 }
fn mk(k: Kind) -> S = S { kind: k, n: 1 }
`

func TestMatchOnAnImportedEnumIsChecked(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "m.tw", kindModule)
	diags := checkFileIn(t, dir, "main.tw", `mode systems
import "m.tw" as m
fn f(s: m.S) -> Str {
  match s.kind {
    Kind.Map => "map",
    Kind.Filter => "filter",
  }
}
`)
	if len(diags) != 1 {
		t.Fatalf("got %d diagnostics, want 1: %v", len(diags), diags)
	}
	const want = "match on Kind is not exhaustive: missing Batch"
	if diags[0].Msg != want {
		t.Errorf("msg = %q, want %q", diags[0].Msg, want)
	}
}

func TestAnExhaustiveMatchOnAnImportedEnumIsQuiet(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "m.tw", kindModule)
	diags := checkFileIn(t, dir, "main.tw", `mode systems
import "m.tw" as m
fn f(s: m.S) -> Str {
  match s.kind {
    Kind.Map => "map",
    Kind.Filter => "filter",
    Kind.Batch => "batch",
  }
}
`)
	if len(diags) != 0 {
		t.Errorf("got %v, want none", diags)
	}
}

// The walk follows a chain, so an enum two imports away is known too.
func TestImportedEnumsAreFoundThroughAChain(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "deep.tw", "mode systems\nenum Colour { Red, Green, Blue }\n")
	writeFile(t, dir, "mid.tw", "mode systems\nimport \"deep.tw\" as d\nfn pick() -> Colour = Red\n")
	diags := checkFileIn(t, dir, "main.tw", `mode systems
import "mid.tw" as mid
fn name(c: Colour) -> Str {
  match c {
    Red => "red",
    Green => "green",
  }
}
`)
	if len(diags) != 1 {
		t.Fatalf("got %d diagnostics, want 1: %v", len(diags), diags)
	}
	if diags[0].Msg != "match on Colour is not exhaustive: missing Blue" {
		t.Errorf("msg = %q", diags[0].Msg)
	}
}

// A file's own declaration wins over an imported one of the same name, so a
// module that shadows an imported enum is judged against its own cases.
func TestAFilesOwnEnumWinsOverAnImportedOne(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "m.tw", kindModule)
	diags := checkFileIn(t, dir, "main.tw", `mode systems
import "m.tw" as m
enum Kind { Map, Filter }
fn f(k: Kind) -> Str {
  match k {
    Map => "map",
    Filter => "filter",
  }
}
`)
	if len(diags) != 0 {
		t.Errorf("got %v, want none: the file's own Kind has two cases", diags)
	}
}

// A case name two enums share is ambiguous as a bare name, and a match using
// one is left unjudged rather than judged against whichever was read last.
func TestASharedCaseNameLeavesTheMatchUnjudged(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "a.tw", "mode systems\nenum First { Alpha, Shared }\n")
	writeFile(t, dir, "b.tw", "mode systems\nenum Second { Shared, Omega }\n")
	diags := checkFileIn(t, dir, "main.tw", `mode systems
import "a.tw" as a
import "b.tw" as b
fn f(x) -> Str {
  match x {
    Shared => "shared",
  }
}
`)
	if len(diags) != 0 {
		t.Errorf("got %v, want none", diags)
	}
}

// An import that cannot be read, or does not parse, is skipped in silence: it
// is the importing file being checked, and the other one's problems are
// reported when it is checked itself.
func TestAnUnreadableImportIsSkipped(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "broken.tw", "mode systems\nfn f( {{{\n")
	for _, src := range []string{
		"mode systems\nimport \"nowhere.tw\" as n\nlet x: I64 = 1\n",
		"mode systems\nimport \"broken.tw\" as b\nlet x: I64 = 1\n",
	} {
		if diags := checkFileIn(t, dir, "main.tw", src); len(diags) != 0 {
			t.Errorf("got %v, want none, for:\n%s", diags, src)
		}
	}
}

// A cycle terminates. Two modules importing each other is legal at run time,
// where each loads once, and the enum walk has to agree.
func TestACycleTerminates(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "x.tw", "mode systems\nimport \"y.tw\" as y\nenum Ex { A, B }\n")
	writeFile(t, dir, "y.tw", "mode systems\nimport \"x.tw\" as x\nenum Why { C, D }\n")
	diags := checkFileIn(t, dir, "main.tw", `mode systems
import "x.tw" as x
fn f(e: Ex) -> Str {
  match e {
    A => "a",
  }
}
`)
	if len(diags) != 1 || diags[0].Msg != "match on Ex is not exhaustive: missing B" {
		t.Fatalf("got %v, want the missing-B diagnostic", diags)
	}
}

// Check itself stays pure: it reads no files, so a program checked without a
// path behaves exactly as it did.
func TestCheckWithoutAPathReadsNothing(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "m.tw", kindModule)
	prog, err := parser.Parse("mode systems\nimport \"m.tw\" as m\nfn f(k) -> Str {\n  match k {\n    Kind.Map => \"map\",\n  }\n}\n")
	if err != nil {
		t.Fatal(err)
	}
	if diags := checker.Check(prog); len(diags) != 0 {
		t.Errorf("Check reported %v; it must not read the import", diags)
	}
}
