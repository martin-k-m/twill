// Command raster runs Raster programs, checks them, or starts a REPL.
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/martin-k-m/raster/internal/checker"
	"github.com/martin-k-m/raster/internal/format"
	"github.com/martin-k-m/raster/internal/interp"
	"github.com/martin-k-m/raster/internal/lexer"
	"github.com/martin-k-m/raster/internal/parser"
	"github.com/martin-k-m/raster/internal/value"
)

const version = "1.0.0"

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		repl()
		return
	}

	switch args[0] {
	case "-v", "--version", "version":
		fmt.Println("Raster", version)
	case "-h", "--help", "help":
		usage()
	case "check":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: raster check <file>")
			os.Exit(2)
		}
		os.Exit(checkOnly(args[1]))
	case "fmt":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: raster fmt <file> [--write]")
			os.Exit(2)
		}
		os.Exit(formatFile(args[1], hasFlag(args, "--write") || hasFlag(args, "-w")))
	case "run":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: raster run <file>")
			os.Exit(2)
		}
		os.Exit(runFile(args[1], !hasFlag(args, "--no-check")))
	case "repl":
		repl()
	default:
		if strings.HasPrefix(args[0], "-") {
			fmt.Fprintf(os.Stderr, "raster: unknown flag %q\n", args[0])
			os.Exit(2)
		}
		os.Exit(runFile(args[0], !hasFlag(args, "--no-check")))
	}
}

func runFile(path string, check bool) int {
	src, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "raster: cannot read file %q\n", path)
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
			fmt.Fprintf(os.Stderr, "raster: %d shape error(s); not running (use --no-check to run anyway)\n", len(diags))
			return 1
		}
	}

	ip := interp.New(nil)
	if err := ip.RunFile(path); err != nil {
		reportError(path, string(src), err)
		return 1
	}
	return 0
}

func checkOnly(path string) int {
	src, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "raster: cannot read file %q\n", path)
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
	src, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "raster: cannot read file %q\n", path)
		return 1
	}
	out, ferr := format.Source(string(src))
	if ferr != nil {
		reportError(path, string(src), ferr)
		return 1
	}
	if write {
		if err := os.WriteFile(path, []byte(out), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "raster: cannot write file %q\n", path)
			return 1
		}
		return 0
	}
	fmt.Print(out)
	return 0
}

func repl() {
	fmt.Printf("Raster %s — a tensor-first, differentiable language.\n", version)
	fmt.Println("Type an expression, or :help / :quit. Multi-line blocks continue")
	fmt.Println("until brackets balance.")
	ip := interp.New(nil)
	sc := bufio.NewScanner(os.Stdin)
	var buf strings.Builder
	fmt.Print("raster> ")
	for sc.Scan() {
		line := sc.Text()
		if buf.Len() == 0 {
			switch strings.TrimSpace(line) {
			case ":quit", ":q":
				return
			case ":help":
				usage()
				fmt.Print("raster> ")
				continue
			case "":
				fmt.Print("raster> ")
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
		fmt.Print("raster> ")
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

func usage() {
	fmt.Printf(`Raster %s

Usage:
  raster <file.ra>            Shape-check, then run a program
  raster run <file.ra>        Same as above
  raster run <file> --no-check   Run without the static shape check
  raster check <file.ra>      Shape-check only, no execution
  raster fmt <file.ra>        Print canonically formatted source
  raster fmt <file> --write   Format the file in place
  raster                      Start the REPL
  raster --version            Print the version

In the REPL: :help, :quit
`, version)
}
