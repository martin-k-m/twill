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

// Describe checks an expression on its own and reports its inferred type and
// shape. The expression is checked in an empty scope, so a name a REPL session
// bound is not visible: this answers what the checker can prove about the text,
// which is what a shape question is usually about.
func Describe(expr string) (Description, error) {
	prog, err := parser.Parse("let __describe = " + expr + "\n")
	if err != nil {
		return Description{}, fmt.Errorf("%s", strings.TrimPrefix(err.Error(), "line 1: "))
	}
	c := newChecker(prog)
	env := c.prelude(prog)
	var t Type = tUnknown{}
	for _, s := range prog.Body {
		if let, ok := s.(*ast.Let); ok && let.Name == "__describe" {
			t = c.inferExpr(let.Value, env)
			continue
		}
		c.inferStmt(s, env)
	}
	// A diagnostic on the expression itself is the answer: a shape mismatch is
	// more useful than the unknown type it would otherwise report.
	for _, d := range c.diags {
		return Description{}, fmt.Errorf("%s", d.Msg)
	}
	return Description{Type: c.typeString(t), Shape: shapeString(c, t)}, nil
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
