package main

import (
	"strings"
	"testing"
)

// The CLI additions in 1.6. Each is small; each answers a question that was
// being answered by guessing.

func TestFlagValueReadsBothSpellings(t *testing.T) {
	args := []string{"test", "std/tests", "--filter", "nn"}
	if got := flagValue(args, "--filter"); got != "nn" {
		t.Errorf("--filter value = %q, want %q", got, "nn")
	}
	if got := flagValue([]string{"test", "--filter=nn"}, "--filter"); got != "nn" {
		t.Errorf("--filter=value = %q, want %q", got, "nn")
	}
	if got := flagValue([]string{"test"}, "--filter"); got != "" {
		t.Errorf("absent --filter = %q, want empty", got)
	}
}

// A flag's value is not a path. Before valueFlags, `twill test --filter nn`
// looked for test files under a directory called "nn".
func TestNonFlagArgsSkipsAFlagsValue(t *testing.T) {
	got := nonFlagArgs([]string{"std/tests", "--filter", "nn", "-v", "examples"})
	want := []string{"std/tests", "examples"}
	if len(got) != len(want) {
		t.Fatalf("paths = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("paths = %q, want %q", got, want)
		}
	}
}

func TestReplQuerySplitsVerbFromExpression(t *testing.T) {
	for _, tc := range []struct{ line, verb, expr string }{
		{":type zeros(2, 3)", ":type", "zeros(2, 3)"},
		{":shape a @ b", ":shape", "a @ b"},
	} {
		verb, expr, ok := replQuery(tc.line)
		if !ok || verb != tc.verb || expr != tc.expr {
			t.Errorf("replQuery(%q) = %q, %q, %v", tc.line, verb, expr, ok)
		}
	}
	// A bare verb has nothing to describe, and an ordinary expression is not a
	// query: both fall through to being evaluated.
	for _, line := range []string{":type", ":shape", "zeros(2, 3)", ":quit"} {
		if _, _, ok := replQuery(line); ok {
			t.Errorf("replQuery(%q) claimed to be a query", line)
		}
	}
}

// `:shape` answers from the checker, so it costs nothing for a value that would
// not fit in memory, and it reports a mismatch rather than an unknown.
func TestDescribeExprAnswersFromTheChecker(t *testing.T) {
	for _, tc := range []struct{ verb, expr, want string }{
		{":shape", "zeros(4, 8) @ zeros(8, 2)", "[4, 2]"},
		{":type", "zeros(4, 8) @ zeros(8, 2)", "a tensor of shape [4, 2]"},
		{":type", `"hello"`, "Str"},
		{":shape", "zeros(2, 3) + zeros(4)", "shape mismatch: [2, 3] vs [4] cannot broadcast"},
	} {
		if got := describeExpr(tc.verb, tc.expr); got != tc.want {
			t.Errorf("%s %s = %q, want %q", tc.verb, tc.expr, got, tc.want)
		}
	}
}

// The checks doctor is made of, asserted one at a time. Its exit code is not,
// because it depends on the environment on purpose: under `go test` the
// running binary is the test binary, so the PATH check correctly reports that
// the twill on PATH is a different file.
func TestDoctorChecksAHealthyInstall(t *testing.T) {
	t.Setenv("TWILL_STD", "")
	for _, r := range []checkResult{doctorBinary(), doctorVersion(), doctorStdLib(), doctorStdOverride(), doctorRuntime()} {
		if !r.ok {
			t.Errorf("%s reported a problem: %s", r.name, r.detail)
		}
	}
	if got := doctorVersion().detail; got != version {
		t.Errorf("version = %q, want %q", got, version)
	}
	if !strings.Contains(doctorStdLib().detail, "loading") {
		t.Errorf("standard library = %q, want it to be loading", doctorStdLib().detail)
	}
}

// A TWILL_STD pointing somewhere unreadable is the mistake doctor exists for.
func TestDoctorWarnsAboutAStdOverride(t *testing.T) {
	t.Setenv("TWILL_STD", "no/such/directory")
	r := doctorStdOverride()
	if r.ok || !strings.Contains(r.detail, "no/such/directory") {
		t.Errorf("doctorStdOverride = %+v, want a warning naming the directory", r)
	}
}
