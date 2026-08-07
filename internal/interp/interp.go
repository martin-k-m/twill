// Package interp is the tree-walking evaluator and standard library for Raster.
package interp

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/martin-k-m/raster/internal/ast"
	"github.com/martin-k-m/raster/internal/parser"
	"github.com/martin-k-m/raster/internal/tensor"
	"github.com/martin-k-m/raster/internal/value"
	"github.com/martin-k-m/raster/std"
)

// defaultSeed makes randomness reproducible by default — a run gives the same
// result every time unless seed(...) changes it. Determinism matters for model
// governance and audit.
const defaultSeed = 1

// stdPrefix reserves a leading "std/" for the standard library that ships
// inside the binary. What follows it is a module name, not a file path, so
// `import "std/nn"` means the same thing from any directory and never reaches
// the filesystem. Every other import path is a file, relative to the importer.
const stdPrefix = "std/"

// stdOverrideEnv points stdPrefix at a directory of .ra files instead of the
// embedded copy, for working on the library itself without rebuilding.
const stdOverrideEnv = "RASTER_STD"

// RuntimeError carries a source line for errors raised during evaluation.
type RuntimeError struct {
	Msg  string
	Line int
}

func (e *RuntimeError) Error() string { return fmt.Sprintf("line %d: %s", e.Line, e.Msg) }

// returnSignal unwinds the stack for a Raster `return`.
type returnSignal struct{ value value.Value }

// srcFrame says where the file currently executing came from. dir is what its
// relative paths resolve against; std marks a standard-library module, which
// lives in the binary and so has no directory of its own.
type srcFrame struct {
	dir string
	std bool
}

// Interp holds global state for a running program.
type Interp struct {
	Global   *value.Env
	out      func(string)
	srcStack []srcFrame
	loaded   map[string]bool // plain imports already loaded
	loading  map[string]bool // namespaced imports currently loading (cycle guard)
	rng      *rand.Rand      // deterministic RNG for randn/rand/seed
}

// New creates an interpreter. If out is nil, output goes to stdout.
func New(out func(string)) *Interp {
	if out == nil {
		out = func(s string) { fmt.Println(s) }
	}
	ip := &Interp{
		Global:  value.NewEnv(nil),
		out:     out,
		loaded:  map[string]bool{},
		loading: map[string]bool{},
		rng:     rand.New(rand.NewSource(defaultSeed)),
	}
	ip.installBuiltins()
	return ip
}

func (ip *Interp) panicf(line int, format string, args ...any) {
	panic(&RuntimeError{Msg: fmt.Sprintf(format, args...), Line: line})
}

func (ip *Interp) currentDir() string {
	if n := len(ip.srcStack); n > 0 && ip.srcStack[n-1].dir != "" {
		return ip.srcStack[n-1].dir
	}
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return wd
}

// inStd reports whether the file currently executing is a standard-library
// module.
func (ip *Interp) inStd() bool {
	n := len(ip.srcStack)
	return n > 0 && ip.srcStack[n-1].std
}

// resolvePath makes a relative path absolute against the running script's
// directory (so file I/O is relative to the source file, not the process cwd).
func (ip *Interp) resolvePath(path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(ip.currentDir(), path)
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
	ip.pushSrc(srcFrame{dir: filepath.Dir(abs)})
	defer ip.popSrc()
	_, runErr := ip.Run(string(src))
	return runErr
}

func (ip *Interp) pushSrc(f srcFrame) { ip.srcStack = append(ip.srcStack, f) }

func (ip *Interp) popSrc() { ip.srcStack = ip.srcStack[:len(ip.srcStack)-1] }

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
		// `for i in range(n)` is the loop people write, and going through the
		// list costs a slice of n elements and n scalars before the first
		// iteration runs. For range(3000000) that is a 48 MB slice the
		// collector then walks repeatedly, which profiling put at the top of a
		// scalar loop's cost.
		//
		// Counting instead allocates nothing up front. Everything else is
		// unchanged: each iteration still gets its own scope and its own scalar,
		// so a closure that captures the loop variable captures what it did
		// before.
		if start, end, step, ok := ip.rangeLoop(st.Iter, env); ok {
			for x := start; (step > 0 && x < end) || (step < 0 && x > end); x += step {
				scope := value.NewEnv(env)
				scope.Define(st.Name, value.Num(float64(x)))
				ip.execBlockIn(st.Body, scope)
			}
			return value.TheUnit
		}
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
	case *ast.TypeDecl:
		// Types are checked statically and erased at runtime.
		return value.TheUnit
	case *ast.UnitDecl:
		// Units are checked statically and erased at runtime.
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

// rangeLoop reads `range(...)` bounds straight off a for-loop's iterable, so the
// loop can count rather than walk a list that was built to be thrown away.
//
// It declines unless the call really is the builtin: a file is free to define
// its own `range`, and quietly running this one instead would be a bug nobody
// could find by reading their own source.
func (ip *Interp) rangeLoop(iter ast.Expr, env *value.Env) (start, end, step int, ok bool) {
	call, isCall := iter.(*ast.Call)
	if !isCall {
		return 0, 0, 0, false
	}
	name, isIdent := call.Callee.(*ast.Ident)
	if !isIdent || name.Name != "range" || len(call.Args) < 1 || len(call.Args) > 3 {
		return 0, 0, 0, false
	}
	// Builtins live in the environment like anything else, so the question is
	// not whether `range` is bound but whether it is still the builtin. A file
	// that defines its own must get its own.
	bound, found := env.Get("range")
	if !found {
		return 0, 0, 0, false
	}
	if b, isBuiltin := bound.(*value.Builtin); !isBuiltin || b.Name != "range" {
		return 0, 0, 0, false
	}

	bounds := make([]int, len(call.Args))
	for i, arg := range call.Args {
		n, isNum := rank0Number(ip.evalExpr(arg, env))
		if !isNum {
			return 0, 0, 0, false
		}
		bounds[i] = int(n)
	}

	step = 1
	switch len(bounds) {
	case 1:
		end = bounds[0]
	case 2:
		start, end = bounds[0], bounds[1]
	case 3:
		start, end, step = bounds[0], bounds[1], bounds[2]
	}
	if step == 0 {
		// The builtin reports this as an error, and it is the builtin's to
		// report. Falling through hands it back.
		return 0, 0, 0, false
	}
	return start, end, step, true
}

func (ip *Interp) iterate(v value.Value, line int) []value.Value {
	switch t := v.(type) {
	case value.Num:
		ip.panicf(line, "can only iterate 1-D tensors")
	case *tensor.Tensor:
		if len(t.Shape) == 1 {
			out := make([]value.Value, len(t.Data))
			for i, x := range t.Data {
				// Indexing a tensor never carried the graph across, so the
				// element is a plain number and stays one.
				out[i] = value.Num(x)
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

func (ip *Interp) doImport(st *ast.Import, env *value.Env) {
	mod, err := ip.loadModule(st.Path)
	if err != nil {
		ip.panicf(st.Line, "%s", err.Error())
	}
	prog, perr := parser.Parse(mod.src)
	if perr != nil {
		ip.panicf(st.Line, "in import %q: %s", st.Path, perr.Error())
	}
	ip.pushSrc(mod.frame)
	defer ip.popSrc()

	if st.Alias != "" {
		// Namespaced import: evaluate into a fresh module scope and bind its
		// definitions as a record under the alias. Guard against cycles.
		if ip.loading[mod.key] {
			return
		}
		ip.loading[mod.key] = true
		defer delete(ip.loading, mod.key)

		// A namespaced module gets its own load-once set, because "already
		// loaded" means "already loaded into this scope" and this scope is new.
		// Sharing the outer one meant a plain import at the top level silently
		// hollowed out any namespace that imported the same module: after
		// `import "std/optim"`, the nested plain import inside nn was skipped as
		// already loaded, so `import "std/nn" as nn` came back without any of
		// optim's names. The fresh map still guards cycles within this module.
		outerLoaded := ip.loaded
		ip.loaded = map[string]bool{}
		defer func() { ip.loaded = outerLoaded }()

		// A module scope tracks definition order, so the namespace record's
		// fields come out in declaration order instead of Go map order.
		modEnv := value.NewModuleEnv(ip.Global)
		for _, s := range prog.Body {
			ip.execStmt(s, modEnv)
		}
		rec := value.NewRecord()
		locals := modEnv.Locals()
		for _, name := range modEnv.LocalNames() {
			rec.Set(name, locals[name])
		}
		env.Define(st.Alias, rec)
		return
	}

	// Plain import: definitions land in the importing scope. Load each module
	// once to keep re-imports and cycles cheap.
	if ip.loaded[mod.key] {
		return
	}
	ip.loaded[mod.key] = true
	for _, s := range prog.Body {
		ip.execStmt(s, env)
	}
}

// module is an import that has been located and read: its source, the key that
// identifies it for the load-once and cycle caches, and the frame to run it in.
type module struct {
	key   string
	src   string
	frame srcFrame
}

// loadModule reads an import. A "std/" path names a module of the embedded
// standard library; anything else is a file path.
func (ip *Interp) loadModule(path string) (module, error) {
	if name, ok := strings.CutPrefix(path, stdPrefix); ok {
		return loadStd(name)
	}
	if ip.inStd() {
		// A std module is embedded, so it has no directory to resolve a
		// relative path against, and an override directory must not be able to
		// pull in code from outside itself.
		return module{}, fmt.Errorf("a standard-library module may only import other std modules, not %q", path)
	}
	abs, err := ip.resolveImport(path)
	if err != nil {
		return module{}, fmt.Errorf("cannot read import %q", path)
	}
	src, err := os.ReadFile(abs)
	if err != nil {
		return module{}, fmt.Errorf("cannot read import %q", path)
	}
	return module{key: abs, src: string(src), frame: srcFrame{dir: filepath.Dir(abs)}}, nil
}

// loadStd reads a standard-library module by name, from the override directory
// if one is configured and from the embedded copy otherwise.
func loadStd(name string) (module, error) {
	if rest, ok := strings.CutSuffix(name, ".ra"); ok {
		return module{}, fmt.Errorf("a standard-library import names a module, not a file: write %q, not %q", stdPrefix+rest, stdPrefix+name)
	}
	if !validStdName(name) {
		return module{}, fmt.Errorf("%q is not a standard-library module name", stdPrefix+name)
	}
	mod := module{key: stdPrefix + name, frame: srcFrame{std: true}}
	if dir := os.Getenv(stdOverrideEnv); dir != "" {
		src, err := os.ReadFile(filepath.Join(dir, name+".ra"))
		if err != nil {
			return module{}, fmt.Errorf("%s is set to %q, which has no module %q", stdOverrideEnv, dir, name)
		}
		mod.src = string(src)
		return mod, nil
	}
	src, ok := std.Read(name)
	if !ok {
		return module{}, fmt.Errorf("no standard-library module %q (the library has %s)", name, strings.Join(std.Names(), ", "))
	}
	mod.src = src
	return mod, nil
}

// validStdName keeps a module name a plain identifier, so it cannot walk out of
// the library into the rest of the filesystem via the override directory.
func validStdName(name string) bool {
	if name == "" {
		return false
	}
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_':
		default:
			return false
		}
	}
	return true
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
		return value.Num(ex.Value)
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
	case *ast.Slice:
		return ip.evalSlice(ex, env)
	case *ast.RecordLit:
		rec := value.NewRecord()
		for _, f := range ex.Fields {
			rec.Set(f.Name, ip.evalExpr(f.Value, env))
		}
		return rec
	case *ast.Field:
		target := ip.evalExpr(ex.Target, env)
		rec, ok := target.(*value.Record)
		if !ok {
			ip.panicf(ex.Line, "cannot access field %q of a non-record", ex.Name)
		}
		v, ok := rec.Get(ex.Name)
		if !ok {
			ip.panicf(ex.Line, "record has no field %q", ex.Name)
		}
		return v
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
		if n, isNum := v.(value.Num); isNum {
			return -n
		}
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

	// Two plain numbers are the whole of an interpreted scalar loop, and going
	// through the tensor engine to add them allocates a rank-0 tensor for the
	// answer that nothing will ever differentiate. Neither operand can be
	// carrying a graph here, because a Num never has one.
	if ln, lIsNum := l.(value.Num); lIsNum {
		if rn, rIsNum := r.(value.Num); rIsNum {
			if res, handled := numArith(op, float64(ln), float64(rn)); handled {
				return res
			}
		}
	}

	lt, lok := value.AsTensor(l)
	rt, rok := value.AsTensor(r)
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

// numArith does scalar arithmetic without building a tensor. It reports false
// for operators whose meaning is not purely scalar (`@`) or that it does not
// know, leaving those to the tensor engine so there is one definition of each.
//
// The results have to match broadcastBinary's element functions exactly, so a
// program cannot tell which path evaluated it.
func numArith(op string, x, y float64) (value.Value, bool) {
	switch op {
	case "+":
		return value.Num(x + y), true
	case "-":
		return value.Num(x - y), true
	case "*":
		return value.Num(x * y), true
	case "/":
		return value.Num(x / y), true
	case "%":
		return value.Num(x - math.Floor(x/y)*y), true
	case "^":
		return value.Num(math.Pow(x, y)), true
	}
	return nil, false
}

// rank0Number reads a value that is a single number of no shape: a plain Num,
// or a rank-0 tensor. A one-element vector is deliberately excluded, because it
// was before Num existed and shape is part of what a tensor means.
func rank0Number(v value.Value) (float64, bool) {
	switch t := v.(type) {
	case value.Num:
		return float64(t), true
	case *tensor.Tensor:
		if t.IsScalar() {
			return t.Data[0], true
		}
	}
	return 0, false
}

func (ip *Interp) compare(op string, l, r value.Value, line int) bool {
	// Scalar comparison covers loop conditions and guards, so it reads the
	// numbers out directly rather than widening either side to a tensor first.
	a, aok := rank0Number(l)
	b, bok := rank0Number(r)
	if aok && bok {
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

// deepEqual is `==` for everything the scalar fast path above does not cover.
// It compares structurally: two separately built lists or records are equal
// when their contents are, which is what makes `a == a` hold for a model's
// parameter tree. Values of different types are never equal, and functions,
// which have no structure worth walking, compare by identity.
func deepEqual(l, r value.Value) bool {
	// A number equals a rank-0 tensor holding it: whether a value went down the
	// unboxed path is an implementation detail, not something `==` may see.
	if ln, ok := l.(value.Num); ok {
		rn, isNum := rank0Number(r)
		return isNum && float64(ln) == rn
	}
	if rn, ok := r.(value.Num); ok {
		ln, isNum := rank0Number(l)
		return isNum && ln == float64(rn)
	}
	switch lv := l.(type) {
	case *tensor.Tensor:
		rv, ok := r.(*tensor.Tensor)
		if !ok || !intsEqual(lv.Shape, rv.Shape) || len(lv.Data) != len(rv.Data) {
			return false
		}
		for i := range lv.Data {
			if lv.Data[i] != rv.Data[i] {
				return false
			}
		}
		return true
	case value.Bool:
		rv, ok := r.(value.Bool)
		return ok && lv == rv
	case value.Str:
		rv, ok := r.(value.Str)
		return ok && lv == rv
	case value.Unit:
		_, ok := r.(value.Unit)
		return ok
	case *value.List:
		rv, ok := r.(*value.List)
		if !ok || len(lv.Items) != len(rv.Items) {
			return false
		}
		for i := range lv.Items {
			if !deepEqual(lv.Items[i], rv.Items[i]) {
				return false
			}
		}
		return true
	case *value.Record:
		rv, ok := r.(*value.Record)
		if !ok || len(lv.Keys) != len(rv.Keys) {
			return false
		}
		// Fields are matched by name, so the order they were declared in does
		// not change the answer.
		for _, k := range lv.Keys {
			other, has := rv.Get(k)
			if !has || !deepEqual(lv.Fields[k], other) {
				return false
			}
		}
		return true
	case *value.Closure:
		rv, ok := r.(*value.Closure)
		return ok && lv == rv
	case *value.Builtin:
		rv, ok := r.(*value.Builtin)
		return ok && lv == rv
	}
	// A native opaque value (a fitted gbm model) also compares by identity.
	// Comparing interfaces whose dynamic type is uncomparable panics, so check
	// the type before trusting ==.
	lt, rt := reflect.TypeOf(l), reflect.TypeOf(r)
	if lt != rt {
		return false
	}
	return lt == nil || (lt.Comparable() && l == r)
}

func intsEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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
	n, ok := rank0Number(idxVal)
	if !ok {
		ip.panicf(ex.Line, "index must be a scalar number")
	}
	idx := int(n)

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

func (ip *Interp) evalSlice(ex *ast.Slice, env *value.Env) value.Value {
	target := ip.evalExpr(ex.Target, env)

	dim0 := -1
	switch t := target.(type) {
	case value.Num:
		ip.panicf(ex.Line, "cannot slice a scalar")
	case *tensor.Tensor:
		if len(t.Shape) == 0 {
			ip.panicf(ex.Line, "cannot slice a scalar")
		}
		dim0 = t.Shape[0]
	case *value.List:
		dim0 = len(t.Items)
	default:
		ip.panicf(ex.Line, "value is not sliceable")
	}

	start := 0
	end := dim0
	if ex.Start != nil {
		start = ip.sliceBound(ex.Start, env, ex.Line)
	}
	if ex.End != nil {
		end = ip.sliceBound(ex.End, env, ex.Line)
	}

	switch t := target.(type) {
	case *tensor.Tensor:
		res, err := tensor.SliceAxis0(t, start, end)
		if err != nil {
			ip.panicf(ex.Line, "%s", err.Error())
		}
		return res
	case *value.List:
		if start < 0 {
			start += dim0
		}
		if end < 0 {
			end += dim0
		}
		if start < 0 || end > dim0 || start > end {
			ip.panicf(ex.Line, "slice [%d:%d] out of range for length %d", start, end, dim0)
		}
		items := make([]value.Value, end-start)
		copy(items, t.Items[start:end])
		return &value.List{Items: items}
	}
	return value.TheUnit
}

func (ip *Interp) sliceBound(e ast.Expr, env *value.Env, line int) int {
	n, ok := rank0Number(ip.evalExpr(e, env))
	if !ok {
		ip.panicf(line, "slice bounds must be scalar numbers")
	}
	return int(n)
}

func (ip *Interp) indexTensor(t *tensor.Tensor, idx, line int) *tensor.Tensor {
	res, err := tensor.IndexAxis0(t, idx)
	if err != nil {
		ip.panicf(line, "%s", err.Error())
	}
	return res
}

func paramNames(params []ast.Param) []string {
	names := make([]string, len(params))
	for i, p := range params {
		names[i] = p.Name
	}
	return names
}
