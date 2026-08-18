package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/twill-lang/twill/internal/checker"
	"github.com/twill-lang/twill/internal/interp"
	"github.com/twill-lang/twill/internal/parser"
	"github.com/twill-lang/twill/std"
)

// `twill doctor` answers the question a bug report starts with: what is
// actually installed here?
//
// It checks the things that are commonly wrong and quiet about it -- a stale
// binary earlier on PATH than the new one, a TWILL_STD left pointing at a
// working copy from last month, a standard library that does not load -- and
// prints what it found either way, because "everything is fine" is the answer
// half the time and is worth being able to see.
//
// It is not required for anything. Nothing else calls it and no other command
// depends on it having been run.

type checkResult struct {
	name   string
	detail string
	// ok is false for something actually wrong. A note that is merely
	// informational sets ok true and says its piece in detail.
	ok bool
}

func doctor() int {
	results := []checkResult{
		doctorBinary(),
		doctorVersion(),
		doctorStdLib(),
		doctorStdOverride(),
		doctorRuntime(),
		doctorPathShadow(),
	}
	bad := 0
	for _, r := range results {
		mark := "ok  "
		if !r.ok {
			mark = "WARN"
			bad++
		}
		fmt.Printf("%s  %-22s %s\n", mark, r.name, r.detail)
	}
	fmt.Println()
	if bad == 0 {
		fmt.Println("twill: nothing to report.")
		return 0
	}
	fmt.Printf("twill: %d thing(s) worth looking at.\n", bad)
	return 1
}

func doctorBinary() checkResult {
	exe, err := os.Executable()
	if err != nil {
		return checkResult{"binary", "cannot determine its own path: " + err.Error(), false}
	}
	return checkResult{"binary", exe, true}
}

func doctorVersion() checkResult {
	return checkResult{"version", version, true}
}

// doctorStdLib loads a module and checks a program against it, which is the
// whole path a real program takes: the embedded source is there, it parses, and
// the checker and evaluator agree it is well formed.
func doctorStdLib() checkResult {
	names := std.Names()
	if len(names) == 0 {
		return checkResult{"standard library", "no modules are embedded in this binary", false}
	}
	src := "import \"std/num\" as num\nlet r = num.trace([[1.0, 0.0], [0.0, 2.0]])\n"
	prog, err := parser.Parse(src)
	if err != nil {
		return checkResult{"standard library", "std/num does not parse: " + err.Error(), false}
	}
	if diags := checker.Check(prog); len(diags) > 0 {
		return checkResult{"standard library", "std/num does not check: " + diags[0].Msg, false}
	}
	ip := interp.New(func(string) {})
	if _, err := ip.Run(src); err != nil {
		return checkResult{"standard library", "std/num does not run: " + err.Error(), false}
	}
	return checkResult{"standard library", fmt.Sprintf("%d modules, embedded, loading", len(names)), true}
}

// doctorStdOverride is the one that catches a real mistake most often. TWILL_STD
// replaces the embedded library wholesale, so a stale one silently gives a
// program a different standard library than the binary shipped with.
func doctorStdOverride() checkResult {
	dir := os.Getenv("TWILL_STD")
	if dir == "" {
		return checkResult{"TWILL_STD", "unset, so the embedded library is in use", true}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return checkResult{"TWILL_STD", fmt.Sprintf("set to %q, which cannot be read: %s", dir, err), false}
	}
	count := 0
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tw") {
			count++
		}
	}
	if count == 0 {
		return checkResult{"TWILL_STD", fmt.Sprintf("set to %q, which holds no .tw modules", dir), false}
	}
	return checkResult{"TWILL_STD",
		fmt.Sprintf("set to %q (%d modules) -- this REPLACES the embedded library", dir, count), false}
}

func doctorRuntime() checkResult {
	return checkResult{"runtime", fmt.Sprintf("%s, %s/%s, %d cpu",
		runtime.Version(), runtime.GOOS, runtime.GOARCH, runtime.NumCPU()), true}
}

// doctorPathShadow looks for another twill earlier on PATH than this one. An
// old binary that answers `twill` while a new one sits in the build directory
// is the reason a fix appears not to have landed.
func doctorPathShadow() checkResult {
	exe, err := os.Executable()
	if err != nil {
		return checkResult{"PATH", "cannot check: " + err.Error(), true}
	}
	exe, _ = filepath.EvalSymlinks(exe)
	names := []string{"twill"}
	if runtime.GOOS == "windows" {
		names = []string{"twill.exe", "twill.cmd", "twill.bat"}
	}
	for _, dir := range filepath.SplitList(os.Getenv("PATH")) {
		for _, name := range names {
			cand := filepath.Join(dir, name)
			info, statErr := os.Stat(cand)
			if statErr != nil || info.IsDir() {
				continue
			}
			resolved, _ := filepath.EvalSymlinks(cand)
			if resolved == "" {
				resolved = cand
			}
			if !strings.EqualFold(resolved, exe) {
				return checkResult{"PATH",
					fmt.Sprintf("%s comes first on PATH and is not this binary", resolved), false}
			}
			return checkResult{"PATH", "this binary is the first twill on PATH", true}
		}
	}
	return checkResult{"PATH", "no twill on PATH (running by explicit path)", true}
}

// versionVerbose is `twill --version --verbose`: the version plus what a bug
// report needs to reproduce a build. Nothing here is computed at run time that
// could be wrong; it is what the toolchain recorded.
func versionVerbose() {
	fmt.Printf("Twill %s\n", version)
	fmt.Printf("  go:        %s\n", runtime.Version())
	fmt.Printf("  target:    %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("  std:       %d modules, embedded\n", len(std.Names()))
	if exe, err := os.Executable(); err == nil {
		fmt.Printf("  binary:    %s\n", exe)
	}
	if dir := os.Getenv("TWILL_STD"); dir != "" {
		fmt.Printf("  TWILL_STD: %s (replaces the embedded library)\n", dir)
	}
}
