package interp_test

import (
	"testing"

	"github.com/twill-lang/twill/internal/format"
	"github.com/twill-lang/twill/internal/parser"
)

// A `mode systems` declaration leads the file the self-hosted compiler is
// written in. The bootstrap does not implement that dialect, but it must not
// choke on the line that names it: a systems-mode file built from features the
// bootstrap already has should parse and run, not fail on its first line.

func TestModeDeclarationIsRecorded(t *testing.T) {
	prog, err := parser.Parse("mode systems\nfn add(a, b) = a + b\n")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if prog.Mode != "systems" {
		t.Fatalf("mode = %q, want %q", prog.Mode, "systems")
	}
	if len(prog.Body) != 1 {
		t.Fatalf("body has %d statements, want 1 (the mode line is not one)", len(prog.Body))
	}
}

func TestNoModeLeavesModeEmpty(t *testing.T) {
	prog, err := parser.Parse("fn add(a, b) = a + b\n")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if prog.Mode != "" {
		t.Fatalf("mode = %q, want empty", prog.Mode)
	}
}

// `mode` is not a keyword, so it stays usable as an ordinary name. Only a
// leading `mode <ident>` is the declaration; `mode` bound as a value is not.
func TestModeIsStillAnOrdinaryNameAfterTheFirstLine(t *testing.T) {
	prog, err := parser.Parse("let mode = 3\nprint(str(mode))\n")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if prog.Mode != "" {
		t.Fatalf("mode = %q, want empty; `let mode` is a binding, not a mode line", prog.Mode)
	}
}

func TestSystemsModeFileRuns(t *testing.T) {
	v, out := run(t, "mode systems\nfn add(a, b) = a + b\nprint(str(add(2, 3)))\n")
	_ = v
	if len(out) != 1 || out[0] != "5" {
		t.Fatalf("output = %v, want [5]", out)
	}
}

func TestFmtPreservesTheModeLine(t *testing.T) {
	src := "mode systems\n\nfn add(a, b) = a + b\n"
	got, err := format.Source(src)
	if err != nil {
		t.Fatalf("format error: %v", err)
	}
	if got != src {
		t.Fatalf("round-trip changed the source:\n--- got ---\n%s\n--- want ---\n%s", got, src)
	}
}
