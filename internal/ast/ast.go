// Package ast defines the Aster syntax tree.
package ast

// Node is any tree node.
type Node interface{ Pos() int }

// Stmt is a statement.
type Stmt interface {
	Node
	stmt()
}

// Expr is an expression.
type Expr interface {
	Node
	expr()
}

// ShapeAnno is an optional tensor-shape annotation on a parameter or return.
// Dims holds each dimension size; a value of -1 means "unknown". An empty
// Dims (len 0) means a scalar (rank-0 tensor).
type ShapeAnno struct {
	Dims []int
}

// Unknown reports whether any dimension is unspecified.
func (s ShapeAnno) Unknown() bool {
	for _, d := range s.Dims {
		if d < 0 {
			return true
		}
	}
	return false
}

// Param is a function parameter with an optional shape annotation.
type Param struct {
	Name  string
	Shape *ShapeAnno // nil when unannotated
}

type Program struct {
	Body []Stmt
}

// --- statements ------------------------------------------------------------

type Let struct {
	Name  string
	Value Expr
	Line  int
}

type FnDecl struct {
	Name   string
	Params []Param
	Ret    *ShapeAnno // nil when unannotated
	Body   Expr       // Block or single expression
	Line   int
}

type Assign struct {
	Name  string
	Value Expr
	Line  int
}

type While struct {
	Cond Expr
	Body *Block
	Line int
}

type For struct {
	Name string
	Iter Expr
	Body *Block
	Line int
}

type Return struct {
	Value Expr // nil for a bare return
	Line  int
}

type Import struct {
	Path string
	Line int
}

type ExprStmt struct {
	X    Expr
	Line int
}

func (s *Let) Pos() int      { return s.Line }
func (s *FnDecl) Pos() int   { return s.Line }
func (s *Assign) Pos() int   { return s.Line }
func (s *While) Pos() int    { return s.Line }
func (s *For) Pos() int      { return s.Line }
func (s *Return) Pos() int   { return s.Line }
func (s *Import) Pos() int   { return s.Line }
func (s *ExprStmt) Pos() int { return s.Line }

func (s *Let) stmt()      {}
func (s *FnDecl) stmt()   {}
func (s *Assign) stmt()   {}
func (s *While) stmt()    {}
func (s *For) stmt()      {}
func (s *Return) stmt()   {}
func (s *Import) stmt()   {}
func (s *ExprStmt) stmt() {}

// --- expressions -----------------------------------------------------------

type NumberLit struct {
	Value float64
	Line  int
}

type StringLit struct {
	Value string
	Line  int
}

type BoolLit struct {
	Value bool
	Line  int
}

type Ident struct {
	Name string
	Line int
}

// TensorLit holds numeric or nested-tensor elements.
type TensorLit struct {
	Elements []Expr
	Line     int
}

type ListLit struct {
	Elements []Expr
	Line     int
}

type Lambda struct {
	Params []Param
	Ret    *ShapeAnno
	Body   Expr
	Line   int
}

type Unary struct {
	Op      string
	Operand Expr
	Line    int
}

type Binary struct {
	Op    string
	Left  Expr
	Right Expr
	Line  int
}

type Call struct {
	Callee Expr
	Args   []Expr
	Line   int
}

type Index struct {
	Target Expr
	Index  Expr
	Line   int
}

type IfExpr struct {
	Cond Expr
	Then *Block
	Else Node // *Block, *IfExpr, or nil
	Line int
}

type Block struct {
	Body []Stmt
	Line int
}

func (e *NumberLit) Pos() int { return e.Line }
func (e *StringLit) Pos() int { return e.Line }
func (e *BoolLit) Pos() int   { return e.Line }
func (e *Ident) Pos() int     { return e.Line }
func (e *TensorLit) Pos() int { return e.Line }
func (e *ListLit) Pos() int   { return e.Line }
func (e *Lambda) Pos() int    { return e.Line }
func (e *Unary) Pos() int     { return e.Line }
func (e *Binary) Pos() int    { return e.Line }
func (e *Call) Pos() int      { return e.Line }
func (e *Index) Pos() int     { return e.Line }
func (e *IfExpr) Pos() int    { return e.Line }
func (e *Block) Pos() int     { return e.Line }

func (e *NumberLit) expr() {}
func (e *StringLit) expr() {}
func (e *BoolLit) expr()   {}
func (e *Ident) expr()     {}
func (e *TensorLit) expr() {}
func (e *ListLit) expr()   {}
func (e *Lambda) expr()    {}
func (e *Unary) expr()     {}
func (e *Binary) expr()    {}
func (e *Call) expr()      {}
func (e *Index) expr()     {}
func (e *IfExpr) expr()    {}
func (e *Block) expr()     {}

// Block is also usable as a statement body; it satisfies Stmt too so blocks
// can appear where statements are expected.
func (e *Block) stmt() {}
