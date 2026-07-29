// Command aster runs Aster programs, checks them, or starts a REPL.
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/martin-k-m/aster/internal/checker"
	"github.com/martin-k-m/aster/internal/interp"
	"github.com/martin-k-m/aster/internal/parser"
	"github.com/martin-k-m/aster/internal/value"
)

const version = "0.2.0"

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		repl()
		return
	}

	switch args[0] {
	case "-v", "--version", "version":
		fmt.Println("Aster", version)
	case "-h", "--help", "help":
		usage()
	case "check":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: aster check <file>")
			os.Exit(2)
		}
		os.Exit(checkOnly(args[1]))
	case "run":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: aster run <file>")
			os.Exit(2)
		}
		os.Exit(runFile(args[1], !hasFlag(args, "--no-check")))
	case "repl":
		repl()
	default:
		// Treat a bare argument as a file to run.
		if strings.HasPrefix(args[0], "-") {
			fmt.Fprintf(os.Stderr, "aster: unknown flag %q\n", args[0])
			os.Exit(2)
		}
		os.Exit(runFile(args[0], !hasFlag(args, "--no-check")))
	}
}

func runFile(path string, check bool) int {
	if check {
		src, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "aster: cannot read file %q\n", path)
			return 1
		}
		prog, perr := parser.Parse(string(src))
		if perr != nil {
			fmt.Fprintln(os.Stderr, "SyntaxError:", perr.Error())
			return 1
		}
		diags := checker.Check(prog)
		if len(diags) > 0 {
			for _, d := range diags {
				fmt.Fprintf(os.Stderr, "%s:%d: shape error: %s\n", path, d.Line, d.Msg)
			}
			fmt.Fprintf(os.Stderr, "aster: %d shape error(s); not running (use --no-check to run anyway)\n", len(diags))
			return 1
		}
	}

	ip := interp.New(nil)
	if err := ip.RunFile(path); err != nil {
		fmt.Fprintln(os.Stderr, errorLabel(err), err.Error())
		return 1
	}
	return 0
}

func checkOnly(path string) int {
	src, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "aster: cannot read file %q\n", path)
		return 1
	}
	prog, perr := parser.Parse(string(src))
	if perr != nil {
		fmt.Fprintln(os.Stderr, "SyntaxError:", perr.Error())
		return 1
	}
	diags := checker.Check(prog)
	if len(diags) == 0 {
		fmt.Printf("%s: no shape problems found\n", path)
		return 0
	}
	for _, d := range diags {
		fmt.Fprintf(os.Stderr, "%s:%d: shape error: %s\n", path, d.Line, d.Msg)
	}
	return 1
}

func repl() {
	fmt.Printf("Aster %s — a tensor-first, differentiable language.\n", version)
	fmt.Println("Type an expression, or :help / :quit.")
	ip := interp.New(nil)
	sc := bufio.NewScanner(os.Stdin)
	fmt.Print("aster> ")
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		switch line {
		case ":quit", ":q":
			return
		case ":help":
			usage()
		case "":
		default:
			v, err := ip.Run(line)
			if err != nil {
				fmt.Fprintln(os.Stderr, errorLabel(err), err.Error())
			} else if _, isUnit := v.(value.Unit); !isUnit && v != nil {
				fmt.Println(value.Format(v))
			}
		}
		fmt.Print("aster> ")
	}
	fmt.Println()
}

func errorLabel(err error) string {
	switch err.(type) {
	case *interp.RuntimeError:
		return "RuntimeError:"
	default:
		return "Error:"
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
	fmt.Printf(`Aster %s

Usage:
  aster <file.ast>        Shape-check, then run a program
  aster run <file.ast>    Same as above
  aster run <file> --no-check   Run without the static shape check
  aster check <file.ast>  Shape-check only, no execution
  aster                   Start the REPL
  aster --version         Print the version

In the REPL: :help, :quit
`, version)
}
