// Package interp is the tree-walking evaluator and standard library for Raster.
package interp

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/martin-k-m/raster/internal/ast"
	"github.com/martin-k-m/raster/internal/parser"
	"github.com/martin-k-m/raster/internal/tensor"
	"github.com/martin-k-m/raster/internal/value"
)

// RuntimeError carries a source line for errors raised during evaluation.
type RuntimeError struct {
	Msg  string
	Line int
}

func (e *RuntimeError) Error() string { return fmt.Sprintf("line %d: %s", e.Line, e.Msg) }

// returnSignal unwinds the stack for a Raster `return`.
type returnSignal struct{ value value.Value }

// Interp holds global state for a running program.
type Interp struct {
	Global   *value.Env
	out      func(string)
	dirStack []string
	loaded   map[string]bool
}

// New creates an interpreter. If out is nil, output goes to stdout.
func New(out func(string)) *Interp {
	if out == nil {
		out = func(s string) { fmt.Println(s) }
	}
	ip := &Interp{
		Global: value.NewEnv(nil),
		out:    out,
		loaded: map[string]bool{},
	}
	ip.installBuiltins()
	return ip
}

func (ip *Interp) panicf(line int, format string, args ...any) {
	panic(&RuntimeError{Msg: fmt.Sprintf(format, args...), Line: line})
}

func (ip *Interp) currentDir() string {
	if len(ip.dirStack) > 0 {
		return ip.dirStack[len(ip.dirStack)-1]
	}
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return wd
}

// Run parses and evaluates source, returning the last value.
func (ip *Interp) Run(src string) (result value.Value, err error) {
	prog, perr := parser.Parse(src)
	if perr != nil {
		return nil, perr
	}
	defer func() {
		if r := recover(); r != nil {
			switch e := r.(type) {
			case *RuntimeError:
				err = e
			case returnSignal:
				result = e.value
			default:
				panic(r)
			}
		}
	}()
	result = value.TheUnit
	for _, s := range prog.Body {
		result = ip.execStmt(s, ip.Global)
	}
	return result, nil
}

// RunFile evaluates a file, resolving imports relative to its directory.
func (ip *Interp) RunFile(path string) error {
	src, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("cannot read file %q", path)
	}
	abs, _ := filepath.Abs(path)
	ip.dirStack = append(ip.dirStack, filepath.Dir(abs))
	defer func() { ip.dirStack = ip.dirStack[:len(ip.dirStack)-1] }()
	_, runErr := ip.Run(string(src))
	return runErr
}

// --- statement execution ---------------------------------------------------

func (ip *Interp) execStmt(s ast.Stmt, env *value.Env) value.Value {
	switch st := s.(type) {
	case *ast.Let:
		env.Define(st.Name, ip.evalExpr(st.Value, env))
		return value.TheUnit
	case *ast.FnDecl:
		env.Define(st.Name, &value.Closure{
			Params: paramNames(st.Params),
			Body:   st.Body,
			Env:    env,
			Name:   st.Name,
		})
		return value.TheUnit
	case *ast.Assign:
		v := ip.evalExpr(st.Value, env)
		if !env.Assign(st.Name, v) {
			ip.panicf(st.Line, "cannot assign to undefined variable %q (use 'let' first)", st.Name)
		}
		return value.TheUnit
	case *ast.While:
		for value.Truthy(ip.evalExpr(st.Cond, env)) {
			ip.execBlockIn(st.Body, value.NewEnv(env))
		}
		return value.TheUnit
	case *ast.For:
		items := ip.iterate(ip.evalExpr(st.Iter, env), st.Line)
		for _, item := range items {
			scope := value.NewEnv(env)
			scope.Define(st.Name, item)
			ip.execBlockIn(st.Body, scope)
		}
		return value.TheUnit
	case *ast.Return:
		var v value.Value = value.TheUnit
		if st.Value != nil {
			v = ip.evalExpr(st.Value, env)
		}
		panic(returnSignal{value: v})
	case *ast.Import:
		ip.doImport(st, env)
		return value.TheUnit
	case *ast.ExprStmt:
		return ip.evalExpr(st.X, env)
	case *ast.Block:
		return ip.execBlockIn(st, value.NewEnv(env))
	default:
		ip.panicf(s.Pos(), "unsupported statement")
		return value.TheUnit
	}
}

func (ip *Interp) execBlockIn(b *ast.Block, scope *value.Env) value.Value {
	var last value.Value = value.TheUnit
	for _, s := range b.Body {
		last = ip.execStmt(s, scope)
	}
	return last
}

func (ip *Interp) iterate(v value.Value, line int) []value.Value {
	switch t := v.(type) {
	case *tensor.Tensor:
		if len(t.Shape) == 1 {
			out := make([]value.Value, len(t.Data))
			for i, x := range t.Data {
				out[i] = tensor.Scalar(x)
			}
			return out
		}
		ip.panicf(line, "can only iterate 1-D tensors")
	case *value.List:
		return t.Items
	}
	ip.panicf(line, "value is not iterable")
	return nil
}

func (ip *Interp) doImport(st *ast.Import, _ *value.Env) {
	abs, err := ip.resolveImport(st.Path)
	if err != nil {
		ip.panicf(st.Line, "cannot read import %q", st.Path)
	}
	if ip.loaded[abs] {
		return
	}
	src, err := os.ReadFile(abs)
	if err != nil {
		ip.panicf(st.Line, "cannot read import %q", st.Path)
	}
	ip.loaded[abs] = true
	prog, perr := parser.Parse(string(src))
	if perr != nil {
		ip.panicf(st.Line, "in import %q: %s", st.Path, perr.Error())
	}
	ip.dirStack = append(ip.dirStack, filepath.Dir(abs))
	defer func() { ip.dirStack = ip.dirStack[:len(ip.dirStack)-1] }()
	// Imported top-level definitions land in the global scope.
	for _, s := range prog.Body {
		ip.execStmt(s, ip.Global)
	}
}

// resolveImport looks for an imported file first relative to the importing
// file, then relative to the working directory.
func (ip *Interp) resolveImport(path string) (string, error) {
	if filepath.IsAbs(path) {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
		return "", os.ErrNotExist
	}
	candidates := []string{filepath.Join(ip.currentDir(), path)}
	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(wd, path))
	}
	for _, cand := range candidates {
		if _, err := os.Stat(cand); err == nil {
			abs, _ := filepath.Abs(cand)
			return abs, nil
		}
	}
	return "", os.ErrNotExist
}

// --- expression evaluation -------------------------------------------------

func (ip *Interp) evalExpr(e ast.Expr, env *value.Env) value.Value {
	switch ex := e.(type) {
	case *ast.NumberLit:
		return tensor.Scalar(ex.Value)
	case *ast.StringLit:
		return value.Str(ex.Value)
	case *ast.BoolLit:
		return value.Bool(ex.Value)
	case *ast.Ident:
		v, ok := env.Get(ex.Name)
		if !ok {
			ip.panicf(ex.Line, "undefined variable %q", ex.Name)
		}
		return v
	case *ast.TensorLit:
		nested := ip.tensorNested(ex.Elements, ex.Line)
		t, err := tensor.FromNested(nested)
		if err != nil {
			ip.panicf(ex.Line, "%s", err.Error())
		}
		return t
	case *ast.ListLit:
		items := make([]value.Value, len(ex.Elements))
		for i, el := range ex.Elements {
			items[i] = ip.evalExpr(el, env)
		}
		return &value.List{Items: items}
	case *ast.Lambda:
		return &value.Closure{Params: paramNames(ex.Params), Body: ex.Body, Env: env, Name: ""}
	case *ast.Unary:
		return ip.evalUnary(ex, env)
	case *ast.Binary:
		return ip.evalBinary(ex, env)
	case *ast.Call:
		return ip.evalCall(ex, env)
	case *ast.Index:
		return ip.evalIndex(ex, env)
	case *ast.IfExpr:
		if value.Truthy(ip.evalExpr(ex.Cond, env)) {
			return ip.execBlockIn(ex.Then, value.NewEnv(env))
		}
		switch alt := ex.Else.(type) {
		case nil:
			return value.TheUnit
		case *ast.Block:
			return ip.execBlockIn(alt, value.NewEnv(env))
		case *ast.IfExpr:
			return ip.evalExpr(alt, env)
		}
		return value.TheUnit
	case *ast.Block:
		return ip.execBlockIn(ex, value.NewEnv(env))
	default:
		ip.panicf(e.Pos(), "unsupported expression")
		return value.TheUnit
	}
}

func (ip *Interp) tensorNested(elements []ast.Expr, line int) []any {
	out := make([]any, len(elements))
	for i, e := range elements {
		switch el := e.(type) {
		case *ast.NumberLit:
			out[i] = el.Value
		case *ast.Unary:
			num, ok := el.Operand.(*ast.NumberLit)
			if el.Op == "-" && ok {
				out[i] = -num.Value
			} else {
				ip.panicf(line, "invalid element in tensor literal")
			}
		case *ast.TensorLit:
			out[i] = ip.tensorNested(el.Elements, line)
		default:
			ip.panicf(line, "invalid element in tensor literal")
		}
	}
	return out
}

func (ip *Interp) evalUnary(ex *ast.Unary, env *value.Env) value.Value {
	v := ip.evalExpr(ex.Operand, env)
	if ex.Op == "-" {
		t, ok := v.(*tensor.Tensor)
		if !ok {
			ip.panicf(ex.Line, "unary '-' expects a number/tensor")
		}
		return tensor.Neg(t)
	}
	return value.Bool(!value.Truthy(v))
}

func (ip *Interp) evalBinary(ex *ast.Binary, env *value.Env) value.Value {
	op := ex.Op
	// Short-circuiting logic.
	if op == "and" || op == "&&" {
		l := ip.evalExpr(ex.Left, env)
		if !value.Truthy(l) {
			return l
		}
		return ip.evalExpr(ex.Right, env)
	}
	if op == "or" || op == "||" {
		l := ip.evalExpr(ex.Left, env)
		if value.Truthy(l) {
			return l
		}
		return ip.evalExpr(ex.Right, env)
	}

	l := ip.evalExpr(ex.Left, env)
	r := ip.evalExpr(ex.Right, env)

	switch op {
	case "==", "!=", "<", "<=", ">", ">=":
		return value.Bool(ip.compare(op, l, r, ex.Line))
	}

	lt, lok := l.(*tensor.Tensor)
	rt, rok := r.(*tensor.Tensor)
	if !lok || !rok {
		ip.panicf(ex.Line, "operator %q expects numbers/tensors", op)
	}

	var res *tensor.Tensor
	var err error
	switch op {
	case "+":
		res, err = tensor.Add(lt, rt)
	case "-":
		res, err = tensor.Sub(lt, rt)
	case "*":
		res, err = tensor.Mul(lt, rt)
	case "/":
		res, err = tensor.Div(lt, rt)
	case "%":
		res, err = tensor.Mod(lt, rt)
	case "@":
		res, err = tensor.MatMul(lt, rt)
	case "^":
		if !rt.IsScalar() {
			ip.panicf(ex.Line, "exponent must be a scalar")
		}
		res = tensor.PowScalar(lt, rt.Data[0])
	default:
		ip.panicf(ex.Line, "unknown operator %q", op)
	}
	if err != nil {
		ip.panicf(ex.Line, "%s", err.Error())
	}
	return res
}

func (ip *Interp) compare(op string, l, r value.Value, line int) bool {
	lt, lok := l.(*tensor.Tensor)
	rt, rok := r.(*tensor.Tensor)
	if lok && rok && lt.IsScalar() && rt.IsScalar() {
		a, b := lt.Data[0], rt.Data[0]
		switch op {
		case "==":
			return a == b
		case "!=":
			return a != b
		case "<":
			return a < b
		case "<=":
			return a <= b
		case ">":
			return a > b
		case ">=":
			return a >= b
		}
	}
	if op == "==" || op == "!=" {
		eq := deepEqual(l, r)
		if op == "==" {
			return eq
		}
		return !eq
	}
	ip.panicf(line, "cannot compare these values with %q", op)
	return false
}

func deepEqual(l, r value.Value) bool {
	lt, lok := l.(*tensor.Tensor)
	rt, rok := r.(*tensor.Tensor)
	if lok && rok {
		if len(lt.Data) != len(rt.Data) {
			return false
		}
		for i := range lt.Data {
			if lt.Data[i] != rt.Data[i] {
				return false
			}
		}
		return true
	}
	switch lv := l.(type) {
	case value.Bool:
		rv, ok := r.(value.Bool)
		return ok && lv == rv
	case value.Str:
		rv, ok := r.(value.Str)
		return ok && lv == rv
	}
	return false
}

func (ip *Interp) evalCall(ex *ast.Call, env *value.Env) value.Value {
	callee := ip.evalExpr(ex.Callee, env)
	args := make([]value.Value, len(ex.Args))
	for i, a := range ex.Args {
		args[i] = ip.evalExpr(a, env)
	}
	return ip.Apply(callee, args, ex.Line)
}

// Apply calls a closure or builtin.
func (ip *Interp) Apply(callee value.Value, args []value.Value, line int) value.Value {
	switch fn := callee.(type) {
	case *value.Builtin:
		if !fn.Variadic && fn.Arity >= 0 && fn.Arity != len(args) {
			ip.panicf(line, "%s expects %d argument(s), got %d", fn.Name, fn.Arity, len(args))
		}
		v, err := fn.Fn(args)
		if err != nil {
			if re, ok := err.(*RuntimeError); ok {
				panic(re)
			}
			ip.panicf(line, "%s", err.Error())
		}
		return v
	case *value.Closure:
		if len(fn.Params) != len(args) {
			name := fn.Name
			if name == "" {
				name = "function"
			}
			ip.panicf(line, "%s expects %d argument(s), got %d", name, len(fn.Params), len(args))
		}
		return ip.callClosure(fn, args)
	default:
		ip.panicf(line, "value is not callable: %s", value.Format(callee))
		return value.TheUnit
	}
}

func (ip *Interp) callClosure(c *value.Closure, args []value.Value) (ret value.Value) {
	scope := value.NewEnv(c.Env)
	for i, p := range c.Params {
		scope.Define(p, args[i])
	}
	defer func() {
		if r := recover(); r != nil {
			if rs, ok := r.(returnSignal); ok {
				ret = rs.value
				return
			}
			panic(r)
		}
	}()
	if blk, ok := c.Body.(*ast.Block); ok {
		return ip.execBlockIn(blk, scope)
	}
	return ip.evalExpr(c.Body, scope)
}

func (ip *Interp) evalIndex(ex *ast.Index, env *value.Env) value.Value {
	target := ip.evalExpr(ex.Target, env)
	idxVal := ip.evalExpr(ex.Index, env)
	it, ok := idxVal.(*tensor.Tensor)
	if !ok || !it.IsScalar() {
		ip.panicf(ex.Line, "index must be a scalar number")
	}
	idx := int(it.Data[0])

	switch t := target.(type) {
	case *tensor.Tensor:
		return ip.indexTensor(t, idx, ex.Line)
	case *value.List:
		if idx < 0 || idx >= len(t.Items) {
			ip.panicf(ex.Line, "list index %d out of range", idx)
		}
		return t.Items[idx]
	}
	ip.panicf(ex.Line, "value is not indexable")
	return value.TheUnit
}

func (ip *Interp) indexTensor(t *tensor.Tensor, idx, line int) *tensor.Tensor {
	if len(t.Shape) == 0 {
		ip.panicf(line, "cannot index a scalar")
	}
	dim0 := t.Shape[0]
	if idx < 0 || idx >= dim0 {
		ip.panicf(line, "tensor index %d out of range [0, %d)", idx, dim0)
	}
	if len(t.Shape) == 1 {
		return tensor.Scalar(t.Data[idx])
	}
	rest := t.Shape[1:]
	stride := 1
	for _, d := range rest {
		stride *= d
	}
	slice := make([]float64, stride)
	copy(slice, t.Data[idx*stride:(idx+1)*stride])
	shape := make([]int, len(rest))
	copy(shape, rest)
	return tensor.New(slice, shape)
}

func paramNames(params []ast.Param) []string {
	names := make([]string, len(params))
	for i, p := range params {
		names[i] = p.Name
	}
	return names
}
