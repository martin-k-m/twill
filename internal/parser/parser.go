// Package parser builds an AST from Raster source using recursive descent
// with a Pratt loop for binary operators.
package parser

import (
	"fmt"
	"strconv"

	"github.com/martin-k-m/raster/internal/ast"
	"github.com/martin-k-m/raster/internal/lexer"
)

var precedence = map[string]int{
	"or": 1, "||": 1,
	"and": 2, "&&": 2,
	"==": 3, "!=": 3,
	"<": 4, "<=": 4, ">": 4, ">=": 4,
	"+": 5, "-": 5,
	"*": 6, "/": 6, "%": 6, "@": 6,
	"^": 7,
}

var rightAssoc = map[string]bool{"^": true}

// Parse tokenizes and parses src into a Program.
func Parse(src string) (*ast.Program, error) {
	toks, err := lexer.Tokenize(src)
	if err != nil {
		return nil, err
	}
	p := &parser{toks: toks}
	return p.parseProgram()
}

type parser struct {
	toks []lexer.Token
	pos  int
}

func (p *parser) peek(o int) lexer.Token {
	idx := p.pos + o
	if idx >= len(p.toks) {
		idx = len(p.toks) - 1
	}
	return p.toks[idx]
}

func (p *parser) next() lexer.Token {
	t := p.toks[p.pos]
	p.pos++
	return t
}

func (p *parser) atEnd() bool { return p.peek(0).Kind == lexer.EOF }

func (p *parser) check(value string) bool {
	t := p.peek(0)
	return (t.Kind == lexer.OP || t.Kind == lexer.PUNCT || t.Kind == lexer.KEYWORD) && t.Value == value
}

func (p *parser) match(value string) bool {
	if p.check(value) {
		p.next()
		return true
	}
	return false
}

func (p *parser) expect(value string) (lexer.Token, error) {
	if p.check(value) {
		return p.next(), nil
	}
	t := p.peek(0)
	return t, p.errf(t, "expected %q but found %q", value, tokenText(t))
}

func (p *parser) errf(t lexer.Token, format string, args ...any) error {
	return &lexer.SyntaxError{Msg: fmt.Sprintf(format, args...), Line: t.Line, Col: t.Col}
}

func (p *parser) skipSeparators() {
	for p.match(";") {
	}
}

func (p *parser) parseProgram() (*ast.Program, error) {
	prog := &ast.Program{}
	p.skipSeparators()
	for !p.atEnd() {
		s, err := p.parseStmt()
		if err != nil {
			return nil, err
		}
		prog.Body = append(prog.Body, s)
		p.skipSeparators()
	}
	return prog, nil
}

// --- statements ------------------------------------------------------------

func (p *parser) parseStmt() (ast.Stmt, error) {
	t := p.peek(0)
	if t.Kind == lexer.KEYWORD {
		switch t.Value {
		case "let":
			return p.parseLet()
		case "fn":
			if p.peek(1).Kind == lexer.IDENT {
				return p.parseFnDecl()
			}
		case "while":
			return p.parseWhile()
		case "for":
			return p.parseFor()
		case "return":
			return p.parseReturn()
		case "import":
			return p.parseImport()
		}
	}

	// Assignment: ident '=' expr (but not '==').
	if t.Kind == lexer.IDENT && p.peek(1).Kind == lexer.OP && p.peek(1).Value == "=" {
		name := p.next().Value
		p.next() // '='
		v, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		return &ast.Assign{Name: name, Value: v, Line: t.Line}, nil
	}

	x, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	return &ast.ExprStmt{X: x, Line: t.Line}, nil
}

func (p *parser) parseLet() (ast.Stmt, error) {
	line := p.next().Line // 'let'
	name, err := p.expectIdent()
	if err != nil {
		return nil, err
	}
	if _, err := p.expect("="); err != nil {
		return nil, err
	}
	v, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	return &ast.Let{Name: name, Value: v, Line: line}, nil
}

func (p *parser) parseFnDecl() (ast.Stmt, error) {
	line := p.next().Line // 'fn'
	name, err := p.expectIdent()
	if err != nil {
		return nil, err
	}
	params, ret, err := p.parseSignature()
	if err != nil {
		return nil, err
	}
	body, err := p.parseFnBody()
	if err != nil {
		return nil, err
	}
	return &ast.FnDecl{Name: name, Params: params, Ret: ret, Body: body, Line: line}, nil
}

func (p *parser) parseWhile() (ast.Stmt, error) {
	line := p.next().Line
	cond, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	body, err := p.parseBlock()
	if err != nil {
		return nil, err
	}
	return &ast.While{Cond: cond, Body: body, Line: line}, nil
}

func (p *parser) parseFor() (ast.Stmt, error) {
	line := p.next().Line
	name, err := p.expectIdent()
	if err != nil {
		return nil, err
	}
	if _, err := p.expect("in"); err != nil {
		return nil, err
	}
	iter, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	body, err := p.parseBlock()
	if err != nil {
		return nil, err
	}
	return &ast.For{Name: name, Iter: iter, Body: body, Line: line}, nil
}

func (p *parser) parseReturn() (ast.Stmt, error) {
	line := p.next().Line
	if p.check("}") || p.check(";") || p.atEnd() {
		return &ast.Return{Value: nil, Line: line}, nil
	}
	v, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	return &ast.Return{Value: v, Line: line}, nil
}

func (p *parser) parseImport() (ast.Stmt, error) {
	line := p.next().Line
	t := p.peek(0)
	if t.Kind != lexer.STRING {
		return nil, p.errf(t, "import expects a string path")
	}
	p.next()
	imp := &ast.Import{Path: t.Value, Line: line}
	// Optional `as name` binds the module's definitions to a namespace record.
	if p.peek(0).Kind == lexer.IDENT && p.peek(0).Value == "as" {
		p.next()
		name, err := p.expectIdent()
		if err != nil {
			return nil, err
		}
		imp.Alias = name
	}
	return imp, nil
}

// --- expressions -----------------------------------------------------------

func (p *parser) parseExpr() (ast.Expr, error) { return p.parseBinary(0) }

func (p *parser) parseBinary(minPrec int) (ast.Expr, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	for {
		t := p.peek(0)
		op := t.Value
		isOp := t.Kind == lexer.OP || (t.Kind == lexer.KEYWORD && (op == "and" || op == "or"))
		if !isOp {
			break
		}
		// A '+' or '-' that starts a new line begins a new statement rather
		// than continuing this expression (so a line like `-mean(x)` is not
		// glued onto the previous line as a subtraction). To continue an
		// expression across lines, end the line with the operator.
		if (op == "+" || op == "-") && p.pos > 0 && t.Line > p.toks[p.pos-1].Line {
			break
		}
		prec, ok := precedence[op]
		if !ok || prec < minPrec {
			break
		}
		p.next()
		nextMin := prec + 1
		if rightAssoc[op] {
			nextMin = prec
		}
		right, err := p.parseBinary(nextMin)
		if err != nil {
			return nil, err
		}
		left = &ast.Binary{Op: op, Left: left, Right: right, Line: t.Line}
	}
	return left, nil
}

func (p *parser) parseUnary() (ast.Expr, error) {
	t := p.peek(0)
	if (t.Kind == lexer.OP && (t.Value == "-" || t.Value == "!")) ||
		(t.Kind == lexer.KEYWORD && t.Value == "not") {
		p.next()
		operand, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &ast.Unary{Op: t.Value, Operand: operand, Line: t.Line}, nil
	}
	return p.parsePostfix()
}

func (p *parser) parsePostfix() (ast.Expr, error) {
	x, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}
	for {
		// A '(' or '[' that starts a new line begins a new expression rather
		// than continuing this one as a call or index. This keeps a line like
		// `[a, b]` from being read as an index on the previous line. Keep the
		// call/index on the same line as its target to chain them.
		if (p.check("(") || p.check("[")) && p.pos > 0 && p.peek(0).Line > p.toks[p.pos-1].Line {
			break
		}
		if p.check(".") {
			line := p.next().Line // '.'
			name, err := p.expectIdent()
			if err != nil {
				return nil, err
			}
			x = &ast.Field{Target: x, Name: name, Line: line}
		} else if p.check("(") {
			line := p.peek(0).Line
			args, err := p.parseArgs()
			if err != nil {
				return nil, err
			}
			x = &ast.Call{Callee: x, Args: args, Line: line}
		} else if p.check("[") {
			line := p.next().Line // '['
			node, err := p.parseIndexOrSlice(x, line)
			if err != nil {
				return nil, err
			}
			x = node
		} else {
			break
		}
	}
	return x, nil
}

// parseIndexOrSlice parses the body of a `[...]` after the '[' is consumed. It
// produces an Index (`t[e]`) or a Slice (`t[a:b]`, with either side optional).
func (p *parser) parseIndexOrSlice(target ast.Expr, line int) (ast.Expr, error) {
	var start, end ast.Expr
	var err error
	if !p.check(":") {
		start, err = p.parseExpr()
		if err != nil {
			return nil, err
		}
	}
	if p.match(":") {
		if !p.check("]") {
			end, err = p.parseExpr()
			if err != nil {
				return nil, err
			}
		}
		if _, err := p.expect("]"); err != nil {
			return nil, err
		}
		return &ast.Slice{Target: target, Start: start, End: end, Line: line}, nil
	}
	if _, err := p.expect("]"); err != nil {
		return nil, err
	}
	return &ast.Index{Target: target, Index: start, Line: line}, nil
}

func (p *parser) parsePrimary() (ast.Expr, error) {
	t := p.peek(0)
	switch t.Kind {
	case lexer.NUMBER:
		p.next()
		v, err := strconv.ParseFloat(t.Value, 64)
		if err != nil {
			return nil, p.errf(t, "invalid number %q", t.Value)
		}
		return &ast.NumberLit{Value: v, Line: t.Line}, nil
	case lexer.STRING:
		p.next()
		return &ast.StringLit{Value: t.Value, Line: t.Line}, nil
	case lexer.KEYWORD:
		switch t.Value {
		case "true":
			p.next()
			return &ast.BoolLit{Value: true, Line: t.Line}, nil
		case "false":
			p.next()
			return &ast.BoolLit{Value: false, Line: t.Line}, nil
		case "if":
			return p.parseIf()
		case "fn":
			return p.parseLambda()
		}
	case lexer.IDENT:
		p.next()
		return &ast.Ident{Name: t.Value, Line: t.Line}, nil
	}

	if p.check("(") {
		p.next()
		inner, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(")"); err != nil {
			return nil, err
		}
		return inner, nil
	}
	if p.check("[") {
		return p.parseTensorOrList()
	}
	if p.check("{") {
		if p.looksLikeRecord() {
			return p.parseRecordLit()
		}
		return p.parseBlock()
	}
	return nil, p.errf(t, "unexpected token %q", tokenText(t))
}

func (p *parser) parseIf() (*ast.IfExpr, error) {
	line := p.next().Line // 'if'
	cond, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	then, err := p.parseBlock()
	if err != nil {
		return nil, err
	}
	var elseBranch ast.Node
	if p.match("else") {
		if p.check("if") {
			elseBranch, err = p.parseIf()
		} else {
			elseBranch, err = p.parseBlock()
		}
		if err != nil {
			return nil, err
		}
	}
	return &ast.IfExpr{Cond: cond, Then: then, Else: elseBranch, Line: line}, nil
}

func (p *parser) parseLambda() (ast.Expr, error) {
	line := p.next().Line // 'fn'
	params, ret, err := p.parseSignature()
	if err != nil {
		return nil, err
	}
	body, err := p.parseFnBody()
	if err != nil {
		return nil, err
	}
	return &ast.Lambda{Params: params, Ret: ret, Body: body, Line: line}, nil
}

func (p *parser) parseTensorOrList() (ast.Expr, error) {
	line := p.next().Line // '['
	var elements []ast.Expr
	if !p.check("]") {
		first, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		elements = append(elements, first)
		for p.match(",") {
			if p.check("]") {
				break
			}
			e, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			elements = append(elements, e)
		}
	}
	if _, err := p.expect("]"); err != nil {
		return nil, err
	}
	isTensor := len(elements) > 0
	for _, e := range elements {
		if !isNumericElem(e) {
			isTensor = false
			break
		}
	}
	if isTensor {
		return &ast.TensorLit{Elements: elements, Line: line}, nil
	}
	return &ast.ListLit{Elements: elements, Line: line}, nil
}

func (p *parser) parseBlock() (*ast.Block, error) {
	open, err := p.expect("{")
	if err != nil {
		return nil, err
	}
	blk := &ast.Block{Line: open.Line}
	p.skipSeparators()
	for !p.check("}") && !p.atEnd() {
		s, err := p.parseStmt()
		if err != nil {
			return nil, err
		}
		blk.Body = append(blk.Body, s)
		p.skipSeparators()
	}
	if _, err := p.expect("}"); err != nil {
		return nil, err
	}
	return blk, nil
}

func (p *parser) parseFnBody() (ast.Expr, error) {
	if p.match("=") {
		return p.parseExpr()
	}
	return p.parseBlock()
}

// looksLikeRecord decides whether a `{` starts a record literal (`{ name: ...`)
// rather than a block. A block never begins with `ident :`.
func (p *parser) looksLikeRecord() bool {
	return p.peek(0).Value == "{" &&
		p.peek(1).Kind == lexer.IDENT &&
		p.peek(2).Kind == lexer.PUNCT && p.peek(2).Value == ":"
}

func (p *parser) parseRecordLit() (ast.Expr, error) {
	line := p.expectPunct("{").Line
	rec := &ast.RecordLit{Line: line}
	for !p.check("}") && !p.atEnd() {
		name, err := p.expectIdent()
		if err != nil {
			return nil, err
		}
		if _, err := p.expect(":"); err != nil {
			return nil, err
		}
		val, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		rec.Fields = append(rec.Fields, ast.RecordField{Name: name, Value: val})
		if !p.match(",") {
			break
		}
	}
	if _, err := p.expect("}"); err != nil {
		return nil, err
	}
	return rec, nil
}

func (p *parser) expectPunct(value string) lexer.Token {
	t, _ := p.expect(value)
	return t
}

// parseSignature parses the parameter list and an optional "-> shape" return.
func (p *parser) parseSignature() ([]ast.Param, *ast.ShapeAnno, error) {
	params, err := p.parseParams()
	if err != nil {
		return nil, nil, err
	}
	var ret *ast.ShapeAnno
	if p.match("->") {
		ret, err = p.parseShapeAnno()
		if err != nil {
			return nil, nil, err
		}
	}
	return params, ret, nil
}

func (p *parser) parseParams() ([]ast.Param, error) {
	if _, err := p.expect("("); err != nil {
		return nil, err
	}
	var params []ast.Param
	if !p.check(")") {
		param, err := p.parseParam()
		if err != nil {
			return nil, err
		}
		params = append(params, param)
		for p.match(",") {
			if p.check(")") {
				break
			}
			param, err := p.parseParam()
			if err != nil {
				return nil, err
			}
			params = append(params, param)
		}
	}
	if _, err := p.expect(")"); err != nil {
		return nil, err
	}
	return params, nil
}

func (p *parser) parseParam() (ast.Param, error) {
	name, err := p.expectIdent()
	if err != nil {
		return ast.Param{}, err
	}
	param := ast.Param{Name: name}
	if p.match(":") {
		shape, err := p.parseShapeAnno()
		if err != nil {
			return ast.Param{}, err
		}
		param.Shape = shape
	}
	return param, nil
}

// parseShapeAnno parses "[d0, d1, ...]" where each dim is an integer or "_"
// (unknown). "[]" is a scalar.
func (p *parser) parseShapeAnno() (*ast.ShapeAnno, error) {
	if _, err := p.expect("["); err != nil {
		return nil, err
	}
	anno := &ast.ShapeAnno{Dims: []ast.Dim{}}
	if !p.check("]") {
		dim, err := p.parseDim()
		if err != nil {
			return nil, err
		}
		anno.Dims = append(anno.Dims, dim)
		for p.match(",") {
			if p.check("]") {
				break
			}
			dim, err := p.parseDim()
			if err != nil {
				return nil, err
			}
			anno.Dims = append(anno.Dims, dim)
		}
	}
	if _, err := p.expect("]"); err != nil {
		return nil, err
	}
	return anno, nil
}

func (p *parser) parseDim() (ast.Dim, error) {
	t := p.peek(0)
	if t.Kind == lexer.NUMBER {
		p.next()
		n, err := strconv.Atoi(t.Value)
		if err != nil || n < 0 {
			return ast.Dim{}, p.errf(t, "shape dimension must be a non-negative integer")
		}
		return ast.ConcreteDim(n), nil
	}
	if t.Kind == lexer.IDENT {
		// "_" is an anonymous unknown dim; any other name is a shape variable.
		p.next()
		if t.Value == "_" {
			return ast.AnonDim(), nil
		}
		return ast.VarDim(t.Value), nil
	}
	return ast.Dim{}, p.errf(t, "expected a dimension size or name")
}

func (p *parser) parseArgs() ([]ast.Expr, error) {
	if _, err := p.expect("("); err != nil {
		return nil, err
	}
	var args []ast.Expr
	if !p.check(")") {
		a, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		args = append(args, a)
		for p.match(",") {
			if p.check(")") {
				break
			}
			a, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			args = append(args, a)
		}
	}
	if _, err := p.expect(")"); err != nil {
		return nil, err
	}
	return args, nil
}

func (p *parser) expectIdent() (string, error) {
	t := p.peek(0)
	if t.Kind == lexer.IDENT {
		p.next()
		return t.Value, nil
	}
	return "", p.errf(t, "expected identifier but found %q", tokenText(t))
}

// isNumericElem reports whether e can appear inside a tensor literal.
func isNumericElem(e ast.Expr) bool {
	switch v := e.(type) {
	case *ast.NumberLit:
		return true
	case *ast.TensorLit:
		return true
	case *ast.Unary:
		if v.Op == "-" {
			_, ok := v.Operand.(*ast.NumberLit)
			return ok
		}
	}
	return false
}

func tokenText(t lexer.Token) string {
	if t.Value != "" {
		return t.Value
	}
	switch t.Kind {
	case lexer.EOF:
		return "end of input"
	default:
		return "?"
	}
}
