package checker

import (
	"fmt"
	"strings"

	"github.com/twill-lang/twill/internal/ast"
	"github.com/twill-lang/twill/internal/parser"
)

// Description is what the REPL's `:type` and `:shape` report about an
// expression: what the checker inferred, without running anything.
//
// Asking by evaluating and printing is not the same question. A shape is a
// property of the expression, and the answer should not depend on being able to
// afford the value: `:shape randn(4096, 4096) @ w` costs nothing here and a
// gigabyte there.
type Description struct {
	Type  string
	Shape string
}

// Describe checks an expression against the builtins and reports its inferred
// type and shape. A name only a surrounding file binds is unknown here: this
// answers what the checker can prove about the text by itself, which is what
// the REPL asks.
func Describe(expr string) (Description, error) {
	return DescribeIn("", expr)
}

// DescribeIn is Describe with a file for context: `context` is checked first,
// so a name it binds is in scope when the expression is. That is what an
// editor's hover needs, because the interesting thing to hover is a local and a
// local has no type outside the file that binds it.
//
// The context is read for its bindings, not for its diagnostics. A file being
// typed is usually mid-edit and half-broken, and refusing to describe anything
// until the whole file is clean would make hover useless exactly when it is
// most wanted.
func DescribeIn(context, expr string) (Description, error) {
	src := context
	if src != "" && !strings.HasSuffix(src, "\n") {
		src += "\n"
	}
	src += "let __describe = " + expr + "\n"

	prog, err := parser.Parse(src)
	if err != nil {
		// It may be the context that does not parse, in which case the
		// expression is still describable on its own.
		if context != "" {
			return DescribeIn("", expr)
		}
		return Description{}, fmt.Errorf("%s", trimPosition(err.Error()))
	}

	c := newChecker(prog)
	env := c.prelude(prog)
	var t Type = tUnknown{}
	found := false
	for _, s := range prog.Body {
		if let, ok := s.(*ast.Let); ok && let.Name == "__describe" {
			// Whatever the context complained about is not this expression's
			// problem, so its diagnostics are dropped before the one that is.
			c.diags = nil
			t = c.inferExpr(let.Value, env)
			found = true
			continue
		}
		c.inferStmt(s, env)
	}
	if !found {
		return Description{}, fmt.Errorf("nothing to describe")
	}
	// A diagnostic on the expression itself is the answer: a shape mismatch is
	// more useful than the unknown type it would otherwise report.
	for _, d := range c.diags {
		return Description{}, fmt.Errorf("%s", d.Msg)
	}
	if _, unknown := t.(tUnknown); unknown {
		return Description{}, fmt.Errorf("no type inferred")
	}
	return Description{Type: c.typeString(t), Shape: shapeString(c, t)}, nil
}

// trimPosition drops a parse error's leading position, which is about the
// synthesised line rather than anything the caller wrote.
func trimPosition(msg string) string {
	if rest, found := strings.CutPrefix(msg, "line "); found {
		if i := strings.Index(rest, ": "); i >= 0 {
			return rest[i+2:]
		}
	}
	return msg
}

// shapeString is the shape half of the answer: a tensor's dimensions, and for
// anything else the type, since "it has no shape" is better said by naming what
// it is instead.
func shapeString(c *checker, t Type) string {
	if tt, ok := t.(tTensor); ok {
		s := dimsString(tt)
		if len(tt.unit) > 0 {
			s += " " + unitString(tt.unit)
		}
		return s
	}
	return c.typeString(t)
}
