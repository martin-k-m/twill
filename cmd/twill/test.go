package main

// Test-runner mode. `twill test [path ...]` discovers the `*_test.tw` files
// under the given paths (or the working directory), runs each on a fresh
// interpreter, and reports one line per file plus a summary, exiting non-zero if
// any file failed.
//
// The reason this exists is that a test file is otherwise a program someone has
// to remember to run. Five repositories each hand-rolled a harness and then
// listed every suite in a CI workflow by hand, which is a failure mode with no
// symptom: a new test file passes because it never ran. Discovery by suffix
// removes the list, so a file named `*_test.tw` is in the suite the moment it
// exists.
//
// A file's verdict follows the harness contract that std/tests/README documents
// and calls greppable: a suite prints one line per check and ends with `OK`, or
// with `FAILED` if any check failed. So a file fails here when it raised an
// error (parse, shape, or runtime) or when its output carries the harness's
// failure marker, and passes otherwise. The runner reads the marker rather than
// an exit code because the harness signals in print, and staying with its
// existing contract means no test file has to change.

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/twill-lang/twill/internal/checker"
	"github.com/twill-lang/twill/internal/interp"
	"github.com/twill-lang/twill/internal/parser"
)

// runTests is the entry point for `twill test`. paths are the files and
// directories to search; when empty it searches the working directory. verbose
// prints every file's captured output, not only a failing one's.
func runTests(paths []string, verbose bool, filter string) int {
	if len(paths) == 0 {
		paths = []string{"."}
	}
	files, err := discoverTestFiles(paths)
	if err != nil {
		fmt.Fprintf(os.Stderr, "twill: %s\n", err)
		return 2
	}
	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "twill: no *_test.tw files found")
		return 2
	}
	// --filter keeps the suites whose path contains the substring. Matching on
	// the path rather than on a test name inside a file is what the runner can
	// honestly promise: a file is the unit it runs, and a name inside one is the
	// harness's business.
	if filter != "" {
		kept := files[:0:0]
		for _, f := range files {
			if strings.Contains(filepath.ToSlash(f), filter) {
				kept = append(kept, f)
			}
		}
		if len(kept) == 0 {
			fmt.Fprintf(os.Stderr, "twill: no test file matches %q (of %d found)\n", filter, len(files))
			return 2
		}
		files = kept
	}

	passed, failed := 0, 0
	for _, f := range files {
		res := runOneTestFile(f)
		if res.ok {
			passed++
			fmt.Printf("ok    %s%s\n", f, res.countSuffix())
		} else {
			failed++
			fmt.Printf("FAIL  %s%s\n", f, res.countSuffix())
		}
		// A failing file's output is the point of the run, so show it; a passing
		// file's is noise unless asked for.
		if verbose || !res.ok {
			fmt.Print(indent(res.output))
			if res.errMsg != "" {
				fmt.Fprintln(os.Stderr, indent(res.errMsg))
			}
		}
	}

	fmt.Println()
	fmt.Printf("%d file(s): %d passed, %d failed\n", len(files), passed, failed)
	if failed > 0 {
		return 1
	}
	return 0
}

// testResult is one file's outcome: whether it passed, its captured output, and
// the pass/fail check counts pulled from the harness summary line when present.
type testResult struct {
	ok      bool
	output  string
	errMsg  string
	checksP int
	checksF int
	counted bool
}

// countSuffix renders the harness check counts for the one-line report, or
// nothing when the file printed no summary (a file that is not a harness suite).
func (r testResult) countSuffix() string {
	if !r.counted {
		return ""
	}
	return fmt.Sprintf("  (%d passed, %d failed)", r.checksP, r.checksF)
}

// runOneTestFile checks and runs a single file, capturing its output, and reads
// the harness contract off the result: an error, or a `FAILED` marker, is a
// failing file.
func runOneTestFile(path string) testResult {
	src, err := os.ReadFile(path)
	if err != nil {
		return testResult{ok: false, errMsg: fmt.Sprintf("cannot read file %q", path)}
	}
	prog, perr := parser.Parse(string(src))
	if perr != nil {
		return testResult{ok: false, errMsg: fmt.Sprintf("parse error: %v", perr)}
	}
	if diags := checker.CheckFile(prog, path); len(diags) > 0 {
		var b strings.Builder
		for _, d := range diags {
			fmt.Fprintf(&b, "%s:%d: shape error: %s\n", path, d.Line, d.Msg)
		}
		return testResult{ok: false, errMsg: strings.TrimRight(b.String(), "\n")}
	}

	// Capture at the OS level rather than through the interpreter's print sink,
	// because a suite that tests std/io writes through write_out, which goes
	// straight to stdout and would otherwise escape the runner -- and a FAILED
	// marker that escapes is a failing suite the runner calls green.
	out, runErr := captureOutput(func() error {
		ip := interp.New(nil)
		_, _, err := ip.RunFileMain(path, []string{"twill"})
		return err
	})
	// A suite that ends in exit(n) is reporting its own verdict, not crashing:
	// zero is a pass and anything else is a failure, and either way the summary
	// it printed is the thing worth showing. A harness calling exit(1) on a red
	// suite is the whole point of having exit, so it must not be reported as an
	// interpreter fault.
	var ex *interp.ExitError
	if errors.As(runErr, &ex) {
		res := testResult{output: out, ok: ex.Code == 0}
		res.readSummary()
		if !res.ok && res.errMsg == "" {
			res.errMsg = fmt.Sprintf("suite exited with status %d", ex.Code)
		}
		return res
	}
	if runErr != nil {
		return testResult{ok: false, output: out, errMsg: fmt.Sprintf("%v", runErr)}
	}

	res := testResult{output: out, ok: true}
	res.readSummary()
	return res
}

// captureOutput redirects stdout and stderr to a pipe for the duration of fn and
// returns everything written to either, so a suite's output is captured whichever
// primitive it printed through. The runner is sequential, so swapping the global
// streams is safe. A goroutine drains the pipe to keep a suite that outfills the
// pipe buffer from blocking on its own print.
func captureOutput(fn func() error) (string, error) {
	origOut, origErr := os.Stdout, os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		return "", fn() // capture unavailable: run anyway rather than skip the suite
	}
	os.Stdout, os.Stderr = w, w
	var buf strings.Builder
	done := make(chan struct{})
	go func() {
		io.Copy(&buf, r)
		close(done)
	}()
	runErr := fn()
	w.Close()
	os.Stdout, os.Stderr = origOut, origErr
	<-done
	r.Close()
	return buf.String(), runErr
}

// readSummary parses the harness's two signals out of the captured output: the
// `<name> passed <p> failed <f>` line and the trailing `FAILED` marker. A file
// with a `FAILED` marker, or a nonzero failed count, did not pass. A file with
// neither (not a harness suite) ran clean and passes.
func (r *testResult) readSummary() {
	for _, line := range strings.Split(r.output, "\n") {
		line = strings.TrimSpace(line)
		if line == "FAILED" {
			r.ok = false
		}
		if p, f, okCounts := parseSummaryLine(line); okCounts {
			r.checksP, r.checksF, r.counted = p, f, true
			if f > 0 {
				r.ok = false
			}
		}
	}
}

// parseSummaryLine reads "<name> passed <p> failed <f>" and returns the two
// counts, or okCounts false when the line is not a summary. The harness prints
// exactly this shape (harness.tw report), so the match is on the two keywords
// with a number after each.
func parseSummaryLine(line string) (passed, failed int, okCounts bool) {
	fields := strings.Fields(line)
	pIdx, fIdx := -1, -1
	for i, w := range fields {
		if w == "passed" {
			pIdx = i
		}
		if w == "failed" {
			fIdx = i
		}
	}
	if pIdx < 0 || fIdx < 0 || pIdx+1 >= len(fields) || fIdx+1 >= len(fields) {
		return 0, 0, false
	}
	p, err1 := parseInt(fields[pIdx+1])
	f, err2 := parseInt(fields[fIdx+1])
	if err1 != nil || err2 != nil {
		return 0, 0, false
	}
	return p, f, true
}

func parseInt(s string) (int, error) {
	var n int
	_, err := fmt.Sscanf(s, "%d", &n)
	return n, err
}

// discoverTestFiles expands each path: a directory is walked for files whose
// name ends in `_test.tw`, and a file is taken as given so a single suite can be
// run by name. The result is sorted and deduplicated so a run is deterministic
// and a file named twice is run once.
func discoverTestFiles(paths []string) ([]string, error) {
	seen := map[string]bool{}
	var files []string
	add := func(p string) {
		p = filepath.Clean(p)
		if !seen[p] {
			seen[p] = true
			files = append(files, p)
		}
	}
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			return nil, fmt.Errorf("cannot read %q", p)
		}
		if !info.IsDir() {
			add(p)
			continue
		}
		err = filepath.WalkDir(p, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !d.IsDir() && strings.HasSuffix(d.Name(), "_test.tw") {
				add(path)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	sort.Strings(files)
	return files, nil
}

// indent shifts a captured block right by two spaces so a file's own output
// reads as nested under its report line, and never collides with the runner's
// own `ok`/`FAIL` column.
func indent(s string) string {
	if s == "" {
		return ""
	}
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	for i, line := range lines {
		lines[i] = "    " + line
	}
	return strings.Join(lines, "\n") + "\n"
}
