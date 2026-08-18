// Package ast defines the Twill syntax tree.
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

// UnitFactor is one `name^exp` term of a unit expression (e.g. USD, year^-1).
type UnitFactor struct {
	Name string
	Exp  int
}

// UnitAnno is a scalar unit expression like `USD`, `USD/year`, or `1/year`,
// stored as a product of factors.
type UnitAnno struct {
	Factors []UnitFactor
}

// Param is a function parameter with an optional annotation: a shape
// (`x: [n, 2]`), a declared record type or unit name (`m: Model`, `p: USD`), or
// a compound unit expression (`r: USD/year`).
type Param struct {
	Name     string
	Shape    *ShapeAnno // non-nil for a shape annotation
	TypeName string     // a bare name: a record type or a unit (resolved by the checker)
	Unit     *UnitAnno  // a compound unit expression (has operators)
}

type Program struct {
	// Mode is the file-level mode named by a leading `mode <name>` declaration,
	// or "" when there is none. `mode systems` selects the systems dialect the
	// self-hosted compiler is written in; the bootstrap records it and runs the
	// features it already has, so a systems-mode file built from those parses
	// and runs rather than failing on the mode line.
	Mode string
	Body []Stmt
}

// --- statements ------------------------------------------------------------

type Let struct {
	Name string
	Unit *UnitAnno // optional unit annotation: `let px: USD/share = ...`
	// A named/qualified/generic type annotation (`let d: Arr[I64] = ...`), or "".
	// Advisory, like a parameter's TypeName; a `.` or `[` after the name marks it
	// unambiguously as a type rather than a unit.
	TypeName string
	Value    Expr
	Line     int
}

type FnDecl struct {
	Name    string
	Params  []Param
	Ret     *ShapeAnno // shape return
	RetUnit *UnitAnno  // unit return (`-> USD`)
	RetType string     // named/qualified type return (`-> Repl`, `-> cp.Caps`); advisory
	Body    Expr       // Block or single expression
	Line    int
}

// Assign is `target = value`, where target is an lvalue: a bare name, a field
// (`obj.f = v`), or an index (`arr[i] = v`), and these compose (`a.d[i] = v`).
type Assign struct {
	Target Expr
	Value  Expr
	Line   int
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

// Break and Continue are the loop-control statements, valid inside a while/for.
type Break struct{ Line int }
type Continue struct{ Line int }

type Import struct {
	Path  string
	Alias string // non-empty for `import "..." as name` (a namespaced module)
	Line  int
}

// UnitDecl declares a base unit: `unit USD`.
type UnitDecl struct {
	Name string
	Line int
}

// TypeDecl declares a record type: `type Name = { field: shape, ... }`.
type TypeDecl struct {
	Name   string
	Fields []TypeField
	Line   int
}

type TypeField struct {
	Name  string
	Shape *ShapeAnno
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
func (s *UnitDecl) Pos() int { return s.Line }
func (s *TypeDecl) Pos() int { return s.Line }
func (s *ExprStmt) Pos() int { return s.Line }

func (s *Let) stmt()        {}
func (s *FnDecl) stmt()     {}
func (s *Assign) stmt()     {}
func (s *While) stmt()      {}
func (s *For) stmt()        {}
func (s *Return) stmt()     {}
func (s *Import) stmt()     {}
func (s *UnitDecl) stmt()   {}
func (s *TypeDecl) stmt()   {}
func (s *EnumDecl) stmt()   {}
func (s *StructDecl) stmt() {}
func (s *ExprStmt) stmt()   {}
func (s *Break) stmt()      {}
func (s *Continue) stmt()   {}

func (s *Break) Pos() int    { return s.Line }
func (s *Continue) Pos() int { return s.Line }

// StructDecl declares a record type: `struct Name { field: Type, ... }`. Field
// types are advisory text; records are structural, so the declaration names a
// type the checker can register without constraining how a record is built.
type StructDecl struct {
	Name   string
	Fields []StructField
	Line   int
}

type StructField struct {
	Name string
	Type string // the field's type name (advisory); may be qualified or generic
}

func (s *StructDecl) Pos() int { return s.Line }

// EnumDecl declares a sum type: `enum Name { Case, Case(Payload), ... }`. Each
// case is a variant with an optional single payload. The payload type is kept
// only as a flag and a name (advisory), since the bootstrap does not check it.
type EnumDecl struct {
	Name     string
	Variants []EnumVariant
	Line     int
}

type EnumVariant struct {
	Name       string
	HasPayload bool
	Payload    string // the payload type name (advisory); "" when no payload
}

func (s *EnumDecl) Pos() int { return s.Line }

// --- expressions -----------------------------------------------------------

type NumberLit struct {
	Value float64
	// Text is the literal as written. An integer literal above 2^53 is not the
	// f64 in Value, and the runtime reads the digits to make an exact I64 of it.
	Text string
	Line int
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
	Params  []Param
	Ret     *ShapeAnno
	RetUnit *UnitAnno
	RetType string // named/qualified type return; advisory, like FnDecl.RetType
	Body    Expr
	Line    int
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
	// TypeName is the name in front of a typed literal, `Point { x: 1.0 }`, or
	// "". Records are structural, so it is advisory: the value is the same record
	// `{ ... }` builds, and the name is kept only so the printer can reproduce it.
	TypeName string
	Fields   []RecordField
	Line     int
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
	Body    []Stmt
	Line    int // line of the opening '{'
	EndLine int // line of the closing '}'
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
func (e *Match) expr()     {}
func (e *Try) expr()       {}
func (e *Block) expr()     {}

// Try is the postfix `?`: it unwraps the success case of a Res/Opt value (the
// payload of `Ok`/`Some`) or, on a failure case (`Err`/`None`), returns that
// value from the enclosing function.
type Try struct {
	Expr Expr
	Line int
}

func (e *Try) Pos() int { return e.Line }

// Match is `match subject { pattern => body, ... }`, an expression whose value
// is the body of the arm whose pattern matched. Arms are tried in order.
type Match struct {
	Subject Expr
	Arms    []MatchArm
	Line    int
}

type MatchArm struct {
	Pattern MatchPattern
	// The arm's body is a statement: an expression, a `return`, an assignment,
	// or a block. Its value (for an expression arm) is the match's value.
	Body Stmt
}

// MatchPattern is `Variant`, `Variant(binding)`, or `_`. Binding is "" when the
// variant carries no payload or the payload is ignored; Wildcard is the `_` arm.
type MatchPattern struct {
	Variant  string
	Binding  string
	Wildcard bool
	Line     int
}

func (e *Match) Pos() int { return e.Line }

// Block is also usable as a statement body; it satisfies Stmt too so blocks
// can appear where statements are expected.
func (e *Block) stmt() {}
