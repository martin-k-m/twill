// Command twill runs Twill programs, checks them, or starts a REPL.
package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/martin-k-m/twill/internal/checker"
	"github.com/martin-k-m/twill/internal/format"
	"github.com/martin-k-m/twill/internal/interp"
	"github.com/martin-k-m/twill/internal/lexer"
	"github.com/martin-k-m/twill/internal/parser"
	"github.com/martin-k-m/twill/internal/value"
)

const version = "1.5.1.1"

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		repl()
		return
	}

	switch args[0] {
	case "-v", "--version", "version":
		fmt.Println("Twill", version)
	case "-h", "--help", "help":
		usage()
	case "check":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: twill check <file>")
			os.Exit(2)
		}
		os.Exit(checkOnly(args[1]))
	case "fmt":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: twill fmt <file> [--write]")
			os.Exit(2)
		}
		os.Exit(formatFile(args[1], hasFlag(args, "--write") || hasFlag(args, "-w")))
	case "run":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: twill run <file>")
			os.Exit(2)
		}
		if hasFlag(args, canonicalDumpFlag) {
			os.Exit(runFileCanonical(args[1], !hasFlag(args, "--no-check")))
		}
		os.Exit(runFile(args[1], !hasFlag(args, "--no-check")))
	case "test":
		os.Exit(runTests(nonFlagArgs(args[1:]), hasFlag(args, "--verbose") || hasFlag(args, "-v")))
	case "repl":
		repl()
	default:
		if strings.HasPrefix(args[0], "-") {
			fmt.Fprintf(os.Stderr, "twill: unknown flag %q\n", args[0])
			os.Exit(2)
		}
		if hasFlag(args, canonicalDumpFlag) {
			os.Exit(runFileCanonical(args[0], !hasFlag(args, "--no-check")))
		}
		os.Exit(runFile(args[0], !hasFlag(args, "--no-check")))
	}
}

func runFile(path string, check bool) int {
	if err := interp.CheckLegacyExt(path); err != nil {
		fmt.Fprintf(os.Stderr, "twill: %s\n", err)
		return 1
	}
	src, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "twill: cannot read file %q\n", path)
		return 1
	}

	if check {
		prog, perr := parser.Parse(string(src))
		if perr != nil {
			reportError(path, string(src), perr)
			return 1
		}
		diags := checker.Check(prog)
		if len(diags) > 0 {
			for _, d := range diags {
				fmt.Fprintf(os.Stderr, "%s:%d: shape error: %s\n", path, d.Line, d.Msg)
				showContext(string(src), d.Line, 0)
			}
			fmt.Fprintf(os.Stderr, "twill: %d shape error(s); not running (use --no-check to run anyway)\n", len(diags))
			return 1
		}
	}

	ip := interp.New(nil)
	result, ranMain, err := ip.RunFileMain(path, scriptArgs(path))
	if err != nil {
		// exit(n) is a status, not a fault, so it carries its own code and prints
		// nothing on the way out.
		var ex *interp.ExitError
		if errors.As(err, &ex) {
			return ex.Code
		}
		reportError(path, string(src), err)
		return 1
	}
	// A systems-mode program's main returns its exit code; a numeric-mode file
	// has no main and its top-level value is not an exit code, so it exits 0.
	if ranMain {
		if n, ok := value.AsNumber(result); ok {
			return int(n)
		}
	}
	return 0
}

// scriptArgs is what a run program sees through the args builtin. It mirrors Go's
// os.Args: element 0 is the program name and the rest are the words after the
// script path, so `twill run cli.tw check foo` hands the program
// ["twill", "check", "foo"] and a self-hosted main that drops the first element
// (as os.Args[1:] does) is left with ["check", "foo"].
func scriptArgs(path string) []string {
	rest := os.Args[1:]
	for i, a := range rest {
		if a == path {
			return append([]string{"twill"}, rest[i+1:]...)
		}
	}
	return []string{"twill"}
}

func checkOnly(path string) int {
	if err := interp.CheckLegacyExt(path); err != nil {
		fmt.Fprintf(os.Stderr, "twill: %s\n", err)
		return 1
	}
	src, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "twill: cannot read file %q\n", path)
		return 1
	}
	prog, perr := parser.Parse(string(src))
	if perr != nil {
		reportError(path, string(src), perr)
		return 1
	}
	diags := checker.Check(prog)
	if len(diags) == 0 {
		fmt.Printf("%s: no shape problems found\n", path)
		return 0
	}
	for _, d := range diags {
		fmt.Fprintf(os.Stderr, "%s:%d: shape error: %s\n", path, d.Line, d.Msg)
		showContext(string(src), d.Line, 0)
	}
	return 1
}

func formatFile(path string, write bool) int {
	if err := interp.CheckLegacyExt(path); err != nil {
		fmt.Fprintf(os.Stderr, "twill: %s\n", err)
		return 1
	}
	src, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "twill: cannot read file %q\n", path)
		return 1
	}
	out, ferr := format.Source(string(src))
	if ferr != nil {
		reportError(path, string(src), ferr)
		return 1
	}
	if write {
		if err := os.WriteFile(path, []byte(out), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "twill: cannot write file %q\n", path)
			return 1
		}
		return 0
	}
	fmt.Print(out)
	return 0
}

func repl() {
	fmt.Printf("Twill %s, a tensor-first differentiable language.\n", version)
	fmt.Println("Type an expression, or :help / :quit. Multi-line blocks continue")
	fmt.Println("until brackets balance.")
	ip := interp.New(nil)
	sc := bufio.NewScanner(os.Stdin)
	var buf strings.Builder
	fmt.Print("twill> ")
	for sc.Scan() {
		line := sc.Text()
		if buf.Len() == 0 {
			switch strings.TrimSpace(line) {
			case ":quit", ":q":
				return
			case ":help":
				usage()
				fmt.Print("twill> ")
				continue
			case "":
				fmt.Print("twill> ")
				continue
			}
		}
		buf.WriteString(line)
		buf.WriteByte('\n')
		if needsMoreInput(buf.String()) {
			fmt.Print("   ...> ")
			continue
		}
		src := buf.String()
		buf.Reset()
		v, err := ip.Run(src)
		if err != nil {
			fmt.Fprintln(os.Stderr, err.Error())
		} else if _, isUnit := v.(value.Unit); !isUnit && v != nil {
			fmt.Println(value.Format(v))
		}
		fmt.Print("twill> ")
	}
	fmt.Println()
}

// needsMoreInput reports whether a REPL buffer has unbalanced brackets (or an
// unterminated string), meaning the user should keep typing.
func needsMoreInput(src string) bool {
	toks, err := lexer.Tokenize(src)
	if err != nil {
		return true // e.g. an unterminated string literal
	}
	depth := 0
	for _, t := range toks {
		if t.Kind != lexer.PUNCT {
			continue
		}
		switch t.Value {
		case "(", "[", "{":
			depth++
		case ")", "]", "}":
			depth--
		}
	}
	return depth > 0
}

// reportError prints an error with a source-line excerpt and a caret.
func reportError(path, src string, err error) {
	switch e := err.(type) {
	case *lexer.SyntaxError:
		fmt.Fprintf(os.Stderr, "%s:%d:%d: syntax error: %s\n", path, e.Line, e.Col, e.Msg)
		showContext(src, e.Line, e.Col)
	case *interp.RuntimeError:
		fmt.Fprintf(os.Stderr, "%s:%d: runtime error: %s\n", path, e.Line, e.Msg)
		showContext(src, e.Line, 0)
	default:
		fmt.Fprintln(os.Stderr, "error:", err.Error())
	}
}

// showContext prints the source line and, when col > 0, a caret beneath it.
func showContext(src string, line, col int) {
	if src == "" || line < 1 {
		return
	}
	lines := strings.Split(src, "\n")
	if line > len(lines) {
		return
	}
	prefix := fmt.Sprintf("  %d | ", line)
	fmt.Fprintf(os.Stderr, "%s%s\n", prefix, lines[line-1])
	if col > 0 {
		fmt.Fprintf(os.Stderr, "%s^\n", strings.Repeat(" ", len(prefix)+col-1))
	}
}

func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

// nonFlagArgs drops the leading-dash words, leaving the positional paths. Used
// by `twill test`, whose arguments are any number of paths in any order with an
// optional -v among them.
func nonFlagArgs(args []string) []string {
	var out []string
	for _, a := range args {
		if !strings.HasPrefix(a, "-") {
			out = append(out, a)
		}
	}
	return out
}

func usage() {
	fmt.Printf(`Twill %s

Usage:
  twill <file.tw>            Shape-check, then run a program
  twill run <file.tw>        Same as above
  twill run <file> --no-check   Run without the static shape check
  twill run <file> --dump=canonical   Run, then dump top-level bindings exactly
  twill check <file.tw>      Shape-check only, no execution
  twill fmt <file.tw>        Print canonically formatted source
  twill fmt <file> --write   Format the file in place
  twill test [path ...]      Run every *_test.tw under the paths (default: .)
  twill test <paths> -v      Show each suite's output, not only failures'
  twill                      Start the REPL
  twill --version            Print the version

In the REPL: :help, :quit
`, version)
}
