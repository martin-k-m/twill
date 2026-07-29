// Package ast defines the Raster syntax tree.
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

// Dim is one dimension of a shape annotation: either a concrete Size (>= 0),
// a named shape variable (Var != ""), or an anonymous unknown (both zero-ish:
// Size < 0 and Var == "", written as `_`).
type Dim struct {
	Size int    // >= 0 for a concrete size, -1 otherwise
	Var  string // non-empty for a named shape variable
}

func ConcreteDim(n int) Dim    { return Dim{Size: n} }
func VarDim(name string) Dim   { return Dim{Size: -1, Var: name} }
func AnonDim() Dim             { return Dim{Size: -1} }
func (d Dim) IsConcrete() bool { return d.Size >= 0 }

// ShapeAnno is an optional tensor-shape annotation on a parameter or return.
// An empty Dims (len 0) means a scalar (rank-0 tensor).
type ShapeAnno struct {
	Dims []Dim
}

// ConcreteDims returns the annotation as plain sizes, with -1 for any
// non-concrete dimension (variable or anonymous).
func (s ShapeAnno) ConcreteDims() []int {
	out := make([]int, len(s.Dims))
	for i, d := range s.Dims {
		if d.IsConcrete() {
			out[i] = d.Size
		} else {
			out[i] = -1
		}
	}
	return out
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
	Path  string
	Alias string // non-empty for `import "..." as name` (a namespaced module)
	Line  int
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

// Slice is target[start:end] along the first axis. Start or End may be nil,
// meaning the beginning or the end respectively.
type Slice struct {
	Target Expr
	Start  Expr // nil = from the beginning
	End    Expr // nil = to the end
	Line   int
}

// RecordLit is a record/struct literal: { name: expr, ... }.
type RecordLit struct {
	Fields []RecordField
	Line   int
}

type RecordField struct {
	Name  string
	Value Expr
}

// Field is record field access: target.name.
type Field struct {
	Target Expr
	Name   string
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
func (e *Slice) Pos() int     { return e.Line }
func (e *RecordLit) Pos() int { return e.Line }
func (e *Field) Pos() int     { return e.Line }
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
func (e *Slice) expr()     {}
func (e *RecordLit) expr() {}
func (e *Field) expr()     {}
func (e *IfExpr) expr()    {}
func (e *Block) expr()     {}

// Block is also usable as a statement body; it satisfies Stmt too so blocks
// can appear where statements are expected.
func (e *Block) stmt() {}
