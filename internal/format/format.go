// Package format pretty-prints an Aster AST back into canonical source. The
// output re-parses to an equivalent program and is idempotent.
package format

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/martin-k-m/twill/internal/ast"
	"github.com/martin-k-m/twill/internal/lexer"
	"github.com/martin-k-m/twill/internal/parser"
)

// Source parses src and returns the canonically formatted text. It refuses
// (with an error) rather than drop any comment it can't place.
func Source(src string) (string, error) {
	prog, comments, err := parser.ParseWithComments(src)
	if err != nil {
		return "", err
	}
	out := format(prog, comments)
	// Safety: the formatted output must carry the same comments as the input.
	_, outComments, err := parser.ParseWithComments(out)
	if err != nil {
		return "", fmt.Errorf("internal: formatted output did not re-parse: %w", err)
	}
	if !sameComments(comments, outComments) {
		return "", fmt.Errorf("cannot format: a comment sits somewhere the formatter can't preserve (e.g. inside an inline block); left unchanged")
	}
	return out, nil
}

func format(prog *ast.Program, comments []lexer.Comment) string {
	p := &printer{trailing: map[int]string{}}
	for _, c := range comments {
		if c.Trailing {
			p.trailing[c.Line] = c.Text
		} else {
			p.own = append(p.own, c)
		}
	}
	sort.SliceStable(p.own, func(i, j int) bool { return p.own[i].Line < p.own[j].Line })
	for _, s := range prog.Body {
		p.stmt(s, 0)
	}
	p.emitLeading(0, 1<<30) // flush trailing own-line comments at end of file
	out := p.b.String()
	if !strings.HasSuffix(out, "\n") {
		out += "\n"
	}
	return out
}

func sameComments(a, b []lexer.Comment) bool {
	if len(a) != len(b) {
		return false
	}
	texts := func(cs []lexer.Comment) []string {
		out := make([]string, len(cs))
		for i, c := range cs {
			out[i] = c.Text
		}
		sort.Strings(out)
		return out
	}
	ta, tb := texts(a), texts(b)
	for i := range ta {
		if ta[i] != tb[i] {
			return false
		}
	}
	return true
}

type printer struct {
	b        strings.Builder
	own      []lexer.Comment // own-line comments, sorted by line
	ownIdx   int
	trailing map[int]string // line -> trailing comment text
}

// emitLeading writes any pending own-line comments that come before beforeLine.
func (p *printer) emitLeading(indent, beforeLine int) {
	for p.ownIdx < len(p.own) && p.own[p.ownIdx].Line < beforeLine {
		p.line(indent, commentText(p.own[p.ownIdx].Text))
		p.ownIdx++
	}
}

func commentText(text string) string {
	if text == "" {
		return "#"
	}
	return "# " + text
}

func (p *printer) line(indent int, text string) {
	p.b.WriteString(strings.Repeat("  ", indent))
	p.b.WriteString(text)
	p.b.WriteByte('\n')
}

// lineC writes a statement line, appending its trailing comment if any.
func (p *printer) lineC(indent int, text string, srcLine int) {
	if t, ok := p.trailing[srcLine]; ok {
		text += "  " + commentText(t)
	}
	p.line(indent, text)
}

func (p *printer) stmt(s ast.Stmt, indent int) {
	p.emitLeading(indent, s.Pos())
	switch st := s.(type) {
	case *ast.Let:
		name := st.Name
		if st.Unit != nil {
			name += ": " + p.unitAnno(st.Unit)
		}
		p.lineC(indent, "let "+name+" = "+p.expr(st.Value), st.Line)
	case *ast.Assign:
		p.lineC(indent, st.Name+" = "+p.expr(st.Value), st.Line)
	case *ast.FnDecl:
		p.fnDecl(st, indent)
	case *ast.While:
		p.blockStmt(indent, "while "+p.expr(st.Cond), st.Body, st.Line)
	case *ast.For:
		p.blockStmt(indent, "for "+st.Name+" in "+p.expr(st.Iter), st.Body, st.Line)
	case *ast.Return:
		if st.Value == nil {
			p.lineC(indent, "return", st.Line)
		} else {
			p.lineC(indent, "return "+p.expr(st.Value), st.Line)
		}
	case *ast.Import:
		if st.Alias != "" {
			p.lineC(indent, "import "+strconv.Quote(st.Path)+" as "+st.Alias, st.Line)
		} else {
			p.lineC(indent, "import "+strconv.Quote(st.Path), st.Line)
		}
	case *ast.TypeDecl:
		fields := make([]string, len(st.Fields))
		for i, f := range st.Fields {
			fields[i] = f.Name + ": " + p.shape(f.Shape)
		}
		p.lineC(indent, "type "+st.Name+" = { "+strings.Join(fields, ", ")+" }", st.Line)
	case *ast.ExprStmt:
		p.lineC(indent, p.expr(st.X), st.Line)
	case *ast.Block:
		for _, inner := range st.Body {
			p.stmt(inner, indent)
		}
	}
}

func (p *printer) fnDecl(fn *ast.FnDecl, indent int) {
	sig := "fn " + fn.Name + "(" + p.params(fn.Params) + ")" + p.retPart(fn.Ret, fn.RetUnit)
	if blk, ok := fn.Body.(*ast.Block); ok {
		p.blockStmt(indent, sig, blk, fn.Line)
		return
	}
	p.lineC(indent, sig+" = "+p.expr(fn.Body), fn.Line)
}

// blockStmt prints `<header> {` ... `}` with the body indented, emitting any
// comments that fall inside the block.
func (p *printer) blockStmt(indent int, header string, blk *ast.Block, headerLine int) {
	p.lineC(indent, header+" {", headerLine)
	for _, s := range blk.Body {
		p.stmt(s, indent+1)
	}
	p.emitLeading(indent+1, blk.EndLine) // comments before the closing brace
	p.line(indent, "}")
}

func (p *printer) params(params []ast.Param) string {
	parts := make([]string, len(params))
	for i, prm := range params {
		s := prm.Name
		switch {
		case prm.Shape != nil:
			s += ": " + p.shape(prm.Shape)
		case prm.TypeName != "":
			s += ": " + prm.TypeName
		case prm.Unit != nil:
			s += ": " + p.unitAnno(prm.Unit)
		}
		parts[i] = s
	}
	return strings.Join(parts, ", ")
}

// retPart prints the `-> ...` return annotation, which is either a shape or a
// unit (never both).
func (p *printer) retPart(r *ast.ShapeAnno, u *ast.UnitAnno) string {
	if r != nil {
		return " -> " + p.shape(r)
	}
	if u != nil {
		return " -> " + p.unitAnno(u)
	}
	return ""
}

// unitAnno renders a unit expression as `USD`, `USD/share`, `USD/year^2`, or
// `1/year`, grouping positive-exponent factors as the numerator and
// negative-exponent factors as the denominator.
func (p *printer) unitAnno(u *ast.UnitAnno) string {
	var num, den []string
	for _, f := range u.Factors {
		switch {
		case f.Exp == 0:
			continue
		case f.Exp == 1:
			num = append(num, f.Name)
		case f.Exp > 0:
			num = append(num, f.Name+"^"+strconv.Itoa(f.Exp))
		case f.Exp == -1:
			den = append(den, f.Name)
		default:
			den = append(den, f.Name+"^"+strconv.Itoa(-f.Exp))
		}
	}
	numS := "1"
	if len(num) > 0 {
		numS = strings.Join(num, "*")
	}
	if len(den) == 0 {
		return numS
	}
	return numS + "/" + strings.Join(den, "*")
}

func (p *printer) shape(s *ast.ShapeAnno) string {
	dims := make([]string, len(s.Dims))
	for i, d := range s.Dims {
		switch {
		case d.IsConcrete():
			dims[i] = strconv.Itoa(d.Size)
		case d.Var != "":
			dims[i] = d.Var
		default:
			dims[i] = "_"
		}
	}
	return "[" + strings.Join(dims, ", ") + "]"
}

// --- expressions -----------------------------------------------------------

func (p *printer) expr(e ast.Expr) string {
	switch ex := e.(type) {
	case *ast.NumberLit:
		return formatNumber(ex.Value)
	case *ast.StringLit:
		return strconv.Quote(ex.Value)
	case *ast.BoolLit:
		if ex.Value {
			return "true"
		}
		return "false"
	case *ast.Ident:
		return ex.Name
	case *ast.TensorLit:
		return "[" + p.exprList(ex.Elements) + "]"
	case *ast.ListLit:
		return "[" + p.exprList(ex.Elements) + "]"
	case *ast.RecordLit:
		parts := make([]string, len(ex.Fields))
		for i, f := range ex.Fields {
			parts[i] = f.Name + ": " + p.expr(f.Value)
		}
		return "{ " + strings.Join(parts, ", ") + " }"
	case *ast.Lambda:
		sig := "fn(" + p.params(ex.Params) + ")" + p.retPart(ex.Ret, ex.RetUnit)
		if blk, ok := ex.Body.(*ast.Block); ok {
			return sig + " " + p.inlineBlock(blk)
		}
		return sig + " = " + p.expr(ex.Body)
	case *ast.Unary:
		if ex.Op == "not" {
			return "not " + p.operand(ex.Operand)
		}
		return ex.Op + p.operand(ex.Operand)
	case *ast.Binary:
		return p.parenChild(ex.Left, ex.Op, false) + " " + ex.Op + " " +
			p.parenChild(ex.Right, ex.Op, true)
	case *ast.Call:
		return p.expr(ex.Callee) + "(" + p.exprList(ex.Args) + ")"
	case *ast.Index:
		return p.expr(ex.Target) + "[" + p.expr(ex.Index) + "]"
	case *ast.Slice:
		lo, hi := "", ""
		if ex.Start != nil {
			lo = p.expr(ex.Start)
		}
		if ex.End != nil {
			hi = p.expr(ex.End)
		}
		return p.expr(ex.Target) + "[" + lo + ":" + hi + "]"
	case *ast.Field:
		return p.expr(ex.Target) + "." + ex.Name
	case *ast.IfExpr:
		return p.ifExpr(ex)
	case *ast.Block:
		return p.inlineBlock(ex)
	}
	return "?"
}

// operand parenthesizes a binary expression under a unary operator so
// `-(a + b)` / `not (a and b)` keep their meaning.
func (p *printer) operand(e ast.Expr) string {
	if _, ok := e.(*ast.Binary); ok {
		return "(" + p.expr(e) + ")"
	}
	return p.expr(e)
}

var precedence = map[string]int{
	"or": 1, "||": 1, "and": 2, "&&": 2,
	"==": 3, "!=": 3, "<": 4, "<=": 4, ">": 4, ">=": 4,
	"+": 5, "-": 5, "*": 6, "/": 6, "%": 6, "@": 6, "^": 7,
}

// parenChild formats a binary operand, adding parentheses only when needed to
// preserve the operator tree's precedence and associativity.
func (p *printer) parenChild(child ast.Expr, parentOp string, isRight bool) string {
	cb, ok := child.(*ast.Binary)
	if !ok {
		return p.expr(child)
	}
	childPrec, parentPrec := precedence[cb.Op], precedence[parentOp]
	need := childPrec < parentPrec
	if childPrec == parentPrec {
		rightAssoc := parentOp == "^"
		if isRight != rightAssoc {
			need = true
		}
	}
	s := p.expr(child)
	if need {
		return "(" + s + ")"
	}
	return s
}

func (p *printer) exprList(es []ast.Expr) string {
	parts := make([]string, len(es))
	for i, e := range es {
		parts[i] = p.expr(e)
	}
	return strings.Join(parts, ", ")
}

// inlineBlock prints a block on one logical line for expression contexts, using
// a nested printer for the body.
func (p *printer) ifExpr(ex *ast.IfExpr) string {
	s := "if " + p.expr(ex.Cond) + " " + p.inlineBlock(ex.Then)
	switch alt := ex.Else.(type) {
	case *ast.Block:
		s += " else " + p.inlineBlock(alt)
	case *ast.IfExpr:
		s += " else " + p.ifExpr(alt)
	}
	return s
}

// inlineBlock renders a block as `{ ... }`. Statements are separated by `;` so
// the result stays a single formatted line inside an expression.
func (p *printer) inlineBlock(blk *ast.Block) string {
	if len(blk.Body) == 0 {
		return "{}"
	}
	sub := &printer{}
	for _, s := range blk.Body {
		sub.stmt(s, 0)
	}
	lines := strings.Split(strings.TrimRight(sub.b.String(), "\n"), "\n")
	for i := range lines {
		lines[i] = strings.TrimSpace(lines[i])
	}
	return "{ " + strings.Join(lines, "; ") + " }"
}

func formatNumber(n float64) string {
	if n == float64(int64(n)) {
		return strconv.FormatInt(int64(n), 10)
	}
	return strconv.FormatFloat(n, 'g', -1, 64)
}
