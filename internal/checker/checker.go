// Package checker performs best-effort static shape analysis of a Raster
// program. It infers tensor shapes where it can and reports a diagnostic only
// when a mismatch is certain (both operand shapes fully known and
// incompatible). Anything it cannot determine is left as Unknown, so dynamic
// code never produces a false alarm.
package checker

import (
	"fmt"

	"github.com/martin-k-m/raster/internal/ast"
)

// Diagnostic is a single shape/type finding.
type Diagnostic struct {
	Msg  string
	Line int
}

func (d Diagnostic) Error() string { return fmt.Sprintf("line %d: %s", d.Line, d.Msg) }

// Check analyses a program and returns any diagnostics found.
func Check(prog *ast.Program) []Diagnostic {
	c := &checker{stack: map[ast.Node]bool{}}
	env := newEnv(nil)
	for name := range builtinNames {
		env.define(name, tBuiltin{name})
	}
	for _, s := range prog.Body {
		c.inferStmt(s, env)
	}
	return c.diags
}

type checker struct {
	diags []Diagnostic
	stack map[ast.Node]bool // user functions currently being inferred
}

func (c *checker) report(line int, format string, args ...any) {
	c.diags = append(c.diags, Diagnostic{Msg: fmt.Sprintf(format, args...), Line: line})
}

// --- types -----------------------------------------------------------------

type Type interface{ isType() }

// tTensor dims: a value of -1 means an unknown size; an empty slice is a scalar.
type tTensor struct{ dims []int }
type tUnknown struct{}
type tBool struct{}
type tStr struct{}
type tUnit struct{}
type tList struct{ elems []Type } // nil elems: unknown contents
type tFn struct {
	node   ast.Node
	params []ast.Param
	ret    *ast.ShapeAnno
	body   ast.Expr
	env    *checkEnv
}
type tBuiltin struct{ name string }

func (tTensor) isType()  {}
func (tUnknown) isType() {}
func (tBool) isType()    {}
func (tStr) isType()     {}
func (tUnit) isType()    {}
func (tList) isType()    {}
func (tFn) isType()      {}
func (tBuiltin) isType() {}

func scalar() tTensor { return tTensor{dims: []int{}} }

func isScalar(t tTensor) bool { return len(t.dims) == 0 }

func fullyKnown(t tTensor) bool {
	for _, d := range t.dims {
		if d < 0 {
			return false
		}
	}
	return true
}

func dimsString(t tTensor) string {
	s := "["
	for i, d := range t.dims {
		if i > 0 {
			s += ", "
		}
		if d < 0 {
			s += "_"
		} else {
			s += fmt.Sprintf("%d", d)
		}
	}
	return s + "]"
}

// --- environment -----------------------------------------------------------

type checkEnv struct {
	vars   map[string]Type
	parent *checkEnv
}

func newEnv(parent *checkEnv) *checkEnv {
	return &checkEnv{vars: map[string]Type{}, parent: parent}
}

func (e *checkEnv) get(name string) (Type, bool) {
	for env := e; env != nil; env = env.parent {
		if t, ok := env.vars[name]; ok {
			return t, true
		}
	}
	return nil, false
}

func (e *checkEnv) define(name string, t Type) { e.vars[name] = t }

func (e *checkEnv) assign(name string, t Type) {
	for env := e; env != nil; env = env.parent {
		if _, ok := env.vars[name]; ok {
			env.vars[name] = t
			return
		}
	}
	e.define(name, t)
}

// --- statements ------------------------------------------------------------

func (c *checker) inferStmt(s ast.Stmt, env *checkEnv) {
	switch st := s.(type) {
	case *ast.Let:
		env.define(st.Name, c.inferExpr(st.Value, env))
	case *ast.FnDecl:
		env.define(st.Name, tFn{node: st, params: st.Params, ret: st.Ret, body: st.Body, env: env})
	case *ast.Assign:
		env.assign(st.Name, c.inferExpr(st.Value, env))
	case *ast.While:
		c.inferExpr(st.Cond, env)
		c.inferBlock(st.Body, newEnv(env))
	case *ast.For:
		iter := c.inferExpr(st.Iter, env)
		scope := newEnv(env)
		scope.define(st.Name, elemType(iter))
		c.inferBlock(st.Body, scope)
	case *ast.Return:
		if st.Value != nil {
			c.inferExpr(st.Value, env)
		}
	case *ast.Import:
		// Imported definitions are resolved at runtime; skip statically.
	case *ast.ExprStmt:
		c.inferExpr(st.X, env)
	case *ast.Block:
		c.inferBlock(st, newEnv(env))
	}
}

func (c *checker) inferBlock(b *ast.Block, env *checkEnv) Type {
	var last Type = tUnit{}
	for _, s := range b.Body {
		if es, ok := s.(*ast.ExprStmt); ok {
			last = c.inferExpr(es.X, env)
		} else {
			c.inferStmt(s, env)
			last = tUnit{}
		}
	}
	return last
}

// elemType returns the element type produced by iterating a value.
func elemType(t Type) Type {
	switch v := t.(type) {
	case tTensor:
		if len(v.dims) == 1 {
			return scalar()
		}
		if len(v.dims) > 1 {
			return tTensor{dims: v.dims[1:]}
		}
	case tList:
		return tUnknown{}
	}
	return tUnknown{}
}

// --- expressions -----------------------------------------------------------

func (c *checker) inferExpr(e ast.Expr, env *checkEnv) Type {
	switch ex := e.(type) {
	case *ast.NumberLit:
		return scalar()
	case *ast.StringLit:
		return tStr{}
	case *ast.BoolLit:
		return tBool{}
	case *ast.Ident:
		if t, ok := env.get(ex.Name); ok {
			return t
		}
		return tUnknown{}
	case *ast.TensorLit:
		return c.inferTensorLit(ex)
	case *ast.ListLit:
		elems := make([]Type, len(ex.Elements))
		for i, el := range ex.Elements {
			elems[i] = c.inferExpr(el, env)
		}
		return tList{elems: elems}
	case *ast.Lambda:
		return tFn{node: ex, params: ex.Params, ret: ex.Ret, body: ex.Body, env: env}
	case *ast.Unary:
		return c.inferUnary(ex, env)
	case *ast.Binary:
		return c.inferBinary(ex, env)
	case *ast.Call:
		return c.inferCall(ex, env)
	case *ast.Index:
		return c.inferIndex(ex, env)
	case *ast.IfExpr:
		c.inferExpr(ex.Cond, env)
		then := c.inferBlock(ex.Then, newEnv(env))
		switch alt := ex.Else.(type) {
		case *ast.Block:
			els := c.inferBlock(alt, newEnv(env))
			return join(then, els)
		case *ast.IfExpr:
			return join(then, c.inferExpr(alt, env))
		default:
			return tUnknown{}
		}
	case *ast.Block:
		return c.inferBlock(ex, newEnv(env))
	}
	return tUnknown{}
}

func (c *checker) inferTensorLit(ex *ast.TensorLit) Type {
	var dims []int
	ok := true
	var walk func(elems []ast.Expr, depth int)
	walk = func(elems []ast.Expr, depth int) {
		if depth == len(dims) {
			dims = append(dims, len(elems))
		} else if dims[depth] != len(elems) {
			ok = false
		}
		for _, el := range elems {
			if inner, isT := el.(*ast.TensorLit); isT {
				walk(inner.Elements, depth+1)
			}
		}
	}
	walk(ex.Elements, 0)
	if !ok {
		c.report(ex.Line, "ragged tensor literal: rows have inconsistent lengths")
		return tUnknown{}
	}
	return tTensor{dims: dims}
}

func (c *checker) inferUnary(ex *ast.Unary, env *checkEnv) Type {
	t := c.inferExpr(ex.Operand, env)
	if ex.Op == "-" {
		if tt, ok := t.(tTensor); ok {
			return tt
		}
		return tUnknown{}
	}
	return tBool{}
}

func (c *checker) inferBinary(ex *ast.Binary, env *checkEnv) Type {
	op := ex.Op
	if op == "and" || op == "or" || op == "&&" || op == "||" {
		c.inferExpr(ex.Left, env)
		c.inferExpr(ex.Right, env)
		return tUnknown{}
	}
	l := c.inferExpr(ex.Left, env)
	r := c.inferExpr(ex.Right, env)

	switch op {
	case "==", "!=", "<", "<=", ">", ">=":
		return tBool{}
	}

	lt, lok := l.(tTensor)
	rt, rok := r.(tTensor)

	// A definite non-tensor operand to arithmetic is a type error.
	if isDefiniteNonTensor(l) || isDefiniteNonTensor(r) {
		c.report(ex.Line, "operator %q needs numbers/tensors", op)
		return tUnknown{}
	}
	if !lok || !rok {
		return tUnknown{}
	}

	switch op {
	case "@":
		res, msg := matmulResult(lt, rt)
		if msg != "" {
			c.report(ex.Line, "%s", msg)
			return tUnknown{}
		}
		return res
	case "^":
		return lt
	default: // + - * / %
		res, msg := elementwiseResult(lt, rt)
		if msg != "" {
			c.report(ex.Line, "%s", msg)
			return tUnknown{}
		}
		return res
	}
}

func (c *checker) inferIndex(ex *ast.Index, env *checkEnv) Type {
	t := c.inferExpr(ex.Target, env)
	c.inferExpr(ex.Index, env)
	switch v := t.(type) {
	case tTensor:
		if isScalar(v) {
			c.report(ex.Line, "cannot index a scalar")
			return tUnknown{}
		}
		if len(v.dims) == 1 {
			return scalar()
		}
		return tTensor{dims: v.dims[1:]}
	case tList:
		return tUnknown{}
	}
	return tUnknown{}
}

// --- calls -----------------------------------------------------------------

func (c *checker) inferCall(ex *ast.Call, env *checkEnv) Type {
	callee := c.inferExpr(ex.Callee, env)
	argTypes := make([]Type, len(ex.Args))
	for i, a := range ex.Args {
		argTypes[i] = c.inferExpr(a, env)
	}

	switch fn := callee.(type) {
	case tBuiltin:
		return c.inferBuiltinCall(fn.name, ex, argTypes)
	case tFn:
		return c.inferUserCall(fn, ex, argTypes)
	case tUnknown:
		return tUnknown{}
	case tList, tBool, tStr, tUnit:
		c.report(ex.Line, "value is not callable")
		return tUnknown{}
	}
	return tUnknown{}
}

func (c *checker) inferUserCall(fn tFn, ex *ast.Call, argTypes []Type) Type {
	if len(fn.params) != len(argTypes) {
		c.report(ex.Line, "function expects %d argument(s), got %d", len(fn.params), len(argTypes))
		return tUnknown{}
	}
	// Check annotated parameters against the supplied arguments.
	scope := newEnv(fn.env)
	for i, p := range fn.params {
		if p.Shape != nil {
			want := tTensor{dims: p.Shape.Dims}
			if got, ok := argTypes[i].(tTensor); ok && fullyKnown(want) && fullyKnown(got) && !shapeMatch(want, got) {
				c.report(ex.Line, "argument %d has shape %s but %q expects %s",
					i+1, dimsString(got), p.Name, dimsString(want))
			}
			scope.define(p.Name, want)
		} else {
			scope.define(p.Name, argTypes[i])
		}
	}
	// Guard against infinite recursion during inference.
	if c.stack[fn.node] {
		if fn.ret != nil {
			return tTensor{dims: fn.ret.Dims}
		}
		return tUnknown{}
	}
	c.stack[fn.node] = true
	var bodyType Type
	if blk, ok := fn.body.(*ast.Block); ok {
		bodyType = c.inferBlock(blk, scope)
	} else {
		bodyType = c.inferExpr(fn.body, scope)
	}
	delete(c.stack, fn.node)

	if fn.ret != nil {
		want := tTensor{dims: fn.ret.Dims}
		if got, ok := bodyType.(tTensor); ok && fullyKnown(want) && fullyKnown(got) && !shapeMatch(want, got) {
			c.report(ex.Line, "function returns %s but its signature declares %s",
				dimsString(got), dimsString(want))
		}
		return want
	}
	return bodyType
}

func (c *checker) inferBuiltinCall(name string, ex *ast.Call, argTypes []Type) Type {
	switch name {
	case "relu", "exp", "log", "sin", "cos", "tanh", "sigmoid", "sqrt", "abs", "square", "softmax":
		if len(argTypes) >= 1 {
			if t, ok := argTypes[0].(tTensor); ok {
				return t
			}
		}
		return tUnknown{}
	case "item", "len":
		return scalar()
	case "sum", "mean", "max", "min":
		return c.reduceResult(ex, argTypes)
	case "argmax", "logsumexp":
		return c.axisReduceResult(ex, argTypes)
	case "maximum", "minimum", "greater", "less", "greater_equal", "less_equal", "equal":
		return c.broadcastTwo(ex, argTypes)
	case "where":
		return c.broadcastWhere(ex, argTypes)
	case "clip":
		if len(argTypes) >= 1 {
			if t, ok := argTypes[0].(tTensor); ok {
				return t
			}
		}
		return tUnknown{}
	case "reshape":
		if len(ex.Args) >= 2 {
			if dims, ok := constShape(ex.Args[1:]); ok {
				return tTensor{dims: dims}
			}
		}
		return tUnknown{}
	case "concat", "fold":
		return tUnknown{}
	case "append", "enumerate":
		return tList{}
	case "scalar":
		return scalar()
	case "pow":
		if len(argTypes) >= 1 {
			if t, ok := argTypes[0].(tTensor); ok {
				return t
			}
		}
		return tUnknown{}
	case "matmul", "dot":
		if len(argTypes) == 2 {
			a, aok := argTypes[0].(tTensor)
			b, bok := argTypes[1].(tTensor)
			if aok && bok {
				res, msg := matmulResult(a, b)
				if msg != "" {
					c.report(ex.Line, "%s", msg)
					return tUnknown{}
				}
				return res
			}
		}
		return tUnknown{}
	case "shape":
		return tList{}
	case "transpose":
		if t, ok := argTypes[0].(tTensor); ok {
			if len(ex.Args) == 1 {
				rev := make([]int, len(t.dims))
				for i := range t.dims {
					rev[i] = t.dims[len(t.dims)-1-i]
				}
				return tTensor{dims: rev}
			}
			axes := make([]int, 0, len(ex.Args)-1)
			for _, a := range ex.Args[1:] {
				ax, ok := constInt(a)
				if !ok {
					return tUnknown{}
				}
				axes = append(axes, ax)
			}
			if len(axes) == len(t.dims) {
				perm := make([]int, len(axes))
				for i, ax := range axes {
					if ax < 0 || ax >= len(t.dims) {
						return tUnknown{}
					}
					perm[i] = t.dims[ax]
				}
				return tTensor{dims: perm}
			}
		}
		return tUnknown{}
	case "zeros", "ones", "randn", "rand":
		if dims, ok := constShape(ex.Args); ok {
			return tTensor{dims: dims}
		}
		return tUnknown{}
	case "fill":
		if len(ex.Args) >= 1 {
			if dims, ok := constShape(ex.Args[1:]); ok {
				return tTensor{dims: dims}
			}
		}
		return tUnknown{}
	case "eye":
		if len(ex.Args) == 1 {
			if n, ok := constInt(ex.Args[0]); ok {
				return tTensor{dims: []int{n, n}}
			}
		}
		return tUnknown{}
	case "range", "list", "map", "zip":
		return tList{}
	case "print":
		return tUnit{}
	case "str":
		return tStr{}
	case "grad", "grads", "value_and_grad", "tensor":
		// These return values whose shape depends on runtime data; treat the
		// result as unknown so downstream code is not falsely flagged.
		return tUnknown{}
	}
	return tUnknown{}
}

// reduceResult handles sum/mean/max/min: no axis reduces to a scalar; a
// constant axis over a known shape removes that dimension.
func (c *checker) reduceResult(ex *ast.Call, argTypes []Type) Type {
	if len(argTypes) == 1 {
		return scalar()
	}
	if len(argTypes) == 2 {
		if t, ok := argTypes[0].(tTensor); ok && len(t.dims) > 0 {
			if ax, ok := constInt(ex.Args[1]); ok {
				if ax < 0 {
					ax += len(t.dims)
				}
				if ax >= 0 && ax < len(t.dims) {
					return tTensor{dims: removeDim(t.dims, ax)}
				}
			}
		}
	}
	return tUnknown{}
}

// axisReduceResult handles argmax/logsumexp, which always reduce one axis
// (default: the last).
func (c *checker) axisReduceResult(ex *ast.Call, argTypes []Type) Type {
	t, ok := argTypes[0].(tTensor)
	if !ok || len(t.dims) == 0 {
		return tUnknown{}
	}
	axis := len(t.dims) - 1
	if len(ex.Args) == 2 {
		ax, ok := constInt(ex.Args[1])
		if !ok {
			return tUnknown{}
		}
		axis = ax
	}
	if axis < 0 {
		axis += len(t.dims)
	}
	if axis < 0 || axis >= len(t.dims) {
		return tUnknown{}
	}
	return tTensor{dims: removeDim(t.dims, axis)}
}

func (c *checker) broadcastTwo(ex *ast.Call, argTypes []Type) Type {
	if len(argTypes) != 2 {
		return tUnknown{}
	}
	a, aok := argTypes[0].(tTensor)
	b, bok := argTypes[1].(tTensor)
	if !aok || !bok {
		return tUnknown{}
	}
	res, msg := elementwiseResult(a, b)
	if msg != "" {
		c.report(ex.Line, "%s", msg)
		return tUnknown{}
	}
	return res
}

func (c *checker) broadcastWhere(ex *ast.Call, argTypes []Type) Type {
	if len(argTypes) != 3 {
		return tUnknown{}
	}
	a, aok := argTypes[1].(tTensor)
	b, bok := argTypes[2].(tTensor)
	if !aok || !bok {
		return tUnknown{}
	}
	res, msg := elementwiseResult(a, b)
	if msg != "" {
		c.report(ex.Line, "%s", msg)
		return tUnknown{}
	}
	return res
}

func removeDim(dims []int, axis int) []int {
	out := make([]int, 0, len(dims)-1)
	out = append(out, dims[:axis]...)
	out = append(out, dims[axis+1:]...)
	return out
}

// --- shape rules -----------------------------------------------------------

func shapeMatch(a, b tTensor) bool {
	if len(a.dims) != len(b.dims) {
		return false
	}
	for i := range a.dims {
		if a.dims[i] != b.dims[i] {
			return false
		}
	}
	return true
}

// broadcastDims applies NumPy broadcasting to two shapes that may contain
// unknown dimensions (-1). It returns the result dims, or ok=false only when a
// mismatch is certain (both dims known, unequal, and neither is 1).
func broadcastDims(a, b tTensor) ([]int, bool) {
	ra, rb := len(a.dims), len(b.dims)
	r := ra
	if rb > r {
		r = rb
	}
	out := make([]int, r)
	for i := 0; i < r; i++ {
		da, db := 1, 1
		aKnown, bKnown := true, true
		if i < ra {
			da = a.dims[ra-1-i]
			aKnown = da >= 0
		}
		if i < rb {
			db = b.dims[rb-1-i]
			bKnown = db >= 0
		}
		var d int
		switch {
		case !aKnown && !bKnown:
			d = -1
		case !aKnown:
			if db == 1 {
				d = -1
			} else {
				d = db
			}
		case !bKnown:
			if da == 1 {
				d = -1
			} else {
				d = da
			}
		case da == db:
			d = da
		case da == 1:
			d = db
		case db == 1:
			d = da
		default:
			return nil, false
		}
		out[r-1-i] = d
	}
	return out, true
}

func elementwiseResult(a, b tTensor) (Type, string) {
	dims, ok := broadcastDims(a, b)
	if !ok {
		return nil, fmt.Sprintf("shape mismatch: %s vs %s cannot broadcast", dimsString(a), dimsString(b))
	}
	return tTensor{dims: dims}, ""
}

func matmulResult(a, b tTensor) (Type, string) {
	a2 := a.dims
	if len(a.dims) == 1 {
		a2 = []int{1, a.dims[0]}
	}
	b2 := b.dims
	if len(b.dims) == 1 {
		b2 = []int{b.dims[0], 1}
	}
	if len(a2) != 2 || len(b2) != 2 {
		return tUnknown{}, ""
	}
	k, k2 := a2[1], b2[0]
	if k >= 0 && k2 >= 0 && k != k2 {
		return nil, fmt.Sprintf("shape mismatch in @: %s @ %s (inner %d != %d)", dimsString(a), dimsString(b), k, k2)
	}
	m, n := a2[0], b2[1]
	switch {
	case len(a.dims) == 1 && len(b.dims) == 1:
		return scalar(), ""
	case len(a.dims) == 1:
		return tTensor{dims: []int{n}}, ""
	case len(b.dims) == 1:
		return tTensor{dims: []int{m}}, ""
	default:
		return tTensor{dims: []int{m, n}}, ""
	}
}

// join returns a if a and b agree, otherwise Unknown.
func join(a, b Type) Type {
	at, aok := a.(tTensor)
	bt, bok := b.(tTensor)
	if aok && bok && shapeMatch(at, bt) {
		return at
	}
	return tUnknown{}
}

func isDefiniteNonTensor(t Type) bool {
	switch t.(type) {
	case tBool, tStr, tUnit, tList, tFn, tBuiltin:
		return true
	}
	return false
}

// --- literal shape extraction ---------------------------------------------

func constInt(e ast.Expr) (int, bool) {
	if n, ok := e.(*ast.NumberLit); ok {
		iv := int(n.Value)
		if float64(iv) == n.Value {
			return iv, true
		}
	}
	return 0, false
}

// constShape reads a shape from integer-literal arguments, or a single list
// literal of integer literals.
func constShape(args []ast.Expr) ([]int, bool) {
	if len(args) == 1 {
		if lst, ok := args[0].(*ast.ListLit); ok {
			dims := make([]int, len(lst.Elements))
			for i, el := range lst.Elements {
				n, ok := constInt(el)
				if !ok {
					return nil, false
				}
				dims[i] = n
			}
			return dims, true
		}
	}
	dims := make([]int, len(args))
	for i, a := range args {
		n, ok := constInt(a)
		if !ok {
			return nil, false
		}
		dims[i] = n
	}
	return dims, true
}

var builtinNames = map[string]bool{
	"print": true, "relu": true, "exp": true, "log": true, "sin": true,
	"cos": true, "tanh": true, "sigmoid": true, "sqrt": true, "sum": true,
	"mean": true, "abs": true, "pow": true, "matmul": true, "dot": true,
	"grad": true, "grads": true, "value_and_grad": true, "map": true, "zip": true,
	"tensor": true, "scalar": true, "zeros": true, "ones": true, "fill": true,
	"randn": true, "rand": true, "eye": true, "transpose": true, "shape": true,
	"len": true, "item": true, "range": true, "list": true, "str": true,
	"square": true, "maximum": true, "minimum": true, "greater": true,
	"less": true, "greater_equal": true, "less_equal": true, "equal": true,
	"where": true, "clip": true, "max": true, "min": true, "argmax": true,
	"softmax": true, "logsumexp": true, "reshape": true, "concat": true,
	"fold": true, "append": true, "enumerate": true,
}
