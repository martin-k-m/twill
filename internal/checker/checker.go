// Package checker performs best-effort static shape analysis of a Raster
// program. It infers tensor shapes where it can and reports a diagnostic only
// when a mismatch is certain (both operand shapes fully known and
// incompatible). Anything it cannot determine is left as Unknown, so dynamic
// code never produces a false alarm.
package checker

import (
	"fmt"
	"sort"
	"strings"

	"github.com/martin-k-m/raster/internal/ast"
	"github.com/martin-k-m/raster/internal/tensor"
)

// Diagnostic is a single shape/type finding.
type Diagnostic struct {
	Msg  string
	Line int
}

// Check analyses a program and returns any diagnostics found.
func Check(prog *ast.Program) []Diagnostic {
	c := &checker{stack: map[ast.Node]bool{}, types: map[string]tRecord{}, units: map[string]bool{}}
	// Register declared record types and units first so forward references resolve.
	for _, s := range prog.Body {
		switch d := s.(type) {
		case *ast.TypeDecl:
			c.registerType(d)
		case *ast.UnitDecl:
			c.units[d.Name] = true
		}
	}
	env := newEnv(nil)
	for name := range builtinNames {
		env.define(name, tBuiltin{name})
	}
	for _, s := range prog.Body {
		c.inferStmt(s, env)
	}
	return dedupe(c.diags)
}

// dedupe removes repeated diagnostics (a body checked at definition and again
// at a call site can report the same finding twice).
func dedupe(diags []Diagnostic) []Diagnostic {
	if len(diags) < 2 {
		return diags
	}
	seen := map[string]bool{}
	out := diags[:0]
	for _, d := range diags {
		key := fmt.Sprintf("%d\x00%s", d.Line, d.Msg)
		if !seen[key] {
			seen[key] = true
			out = append(out, d)
		}
	}
	return out
}

type checker struct {
	diags []Diagnostic
	stack map[ast.Node]bool  // user functions currently being inferred
	types map[string]tRecord // declared record types
	units map[string]bool    // declared base units
}

func (c *checker) registerType(td *ast.TypeDecl) {
	fields := map[string]Type{}
	for _, f := range td.Fields {
		fields[f.Name] = tTensor{dims: f.Shape.ConcreteDims()}
	}
	c.types[td.Name] = tRecord{fields: fields}
}

func (c *checker) report(line int, format string, args ...any) {
	c.diags = append(c.diags, Diagnostic{Msg: fmt.Sprintf(format, args...), Line: line})
}

// --- types -----------------------------------------------------------------

type Type interface{ isType() }

// unitMap is a physical unit as base-name -> integer exponent. A nil/empty map
// is dimensionless. Units are checked statically and erased at runtime.
type unitMap map[string]int

// tTensor dims: a value of -1 means an unknown size; an empty slice is a scalar.
// unit is the quantity's physical unit (nil = dimensionless).
type tTensor struct {
	dims []int
	unit unitMap
}
type tUnknown struct{}
type tBool struct{}
type tStr struct{}
type tUnit struct{}
type tList struct{ elems []Type } // nil elems: unknown contents
type tRecord struct{ fields map[string]Type }
type tFn struct {
	node    ast.Node
	params  []ast.Param
	ret     *ast.ShapeAnno
	retUnit *ast.UnitAnno
	body    ast.Expr
	env     *checkEnv
}
type tBuiltin struct{ name string }

func (tTensor) isType()  {}
func (tUnknown) isType() {}
func (tBool) isType()    {}
func (tStr) isType()     {}
func (tUnit) isType()    {}
func (tList) isType()    {}
func (tRecord) isType()  {}
func (tFn) isType()      {}
func (tBuiltin) isType() {}

func scalar() tTensor { return tTensor{dims: []int{}} }

// --- units ------------------------------------------------------------------

func (u unitMap) clean() unitMap {
	for k, v := range u {
		if v == 0 {
			delete(u, k)
		}
	}
	if len(u) == 0 {
		return nil
	}
	return u
}

func unitEqual(a, b unitMap) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

func unitMul(a, b unitMap) unitMap {
	r := unitMap{}
	for k, v := range a {
		r[k] += v
	}
	for k, v := range b {
		r[k] += v
	}
	return r.clean()
}

func unitDiv(a, b unitMap) unitMap {
	r := unitMap{}
	for k, v := range a {
		r[k] += v
	}
	for k, v := range b {
		r[k] -= v
	}
	return r.clean()
}

func unitPow(a unitMap, k int) unitMap {
	if k == 0 || len(a) == 0 {
		return nil
	}
	r := unitMap{}
	for name, v := range a {
		r[name] = v * k
	}
	return r.clean()
}

// unitSqrt halves each exponent; ok is false if any is odd.
func unitSqrt(a unitMap) (unitMap, bool) {
	r := unitMap{}
	for name, v := range a {
		if v%2 != 0 {
			return nil, false
		}
		r[name] = v / 2
	}
	return r.clean(), true
}

func unitString(u unitMap) string {
	if len(u) == 0 {
		return "1"
	}
	names := make([]string, 0, len(u))
	for k := range u {
		names = append(names, k)
	}
	sort.Strings(names)
	parts := make([]string, len(names))
	for i, n := range names {
		if u[n] == 1 {
			parts[i] = n
		} else {
			parts[i] = fmt.Sprintf("%s^%d", n, u[n])
		}
	}
	return strings.Join(parts, "*")
}

func unitFromAnno(a *ast.UnitAnno) unitMap {
	r := unitMap{}
	for _, f := range a.Factors {
		r[f.Name] += f.Exp
	}
	return r.clean()
}

// resolveUnit builds a unit map from an annotation and reports any factor that
// names a unit which was never declared with `unit NAME`.
func (c *checker) resolveUnit(a *ast.UnitAnno, line int) unitMap {
	for _, f := range a.Factors {
		if !c.units[f.Name] {
			c.report(line, "unknown unit %q (declare it with `unit %s`)", f.Name, f.Name)
		}
	}
	return unitFromAnno(a)
}

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
		rhs := c.inferExpr(st.Value, env)
		if st.Unit != nil {
			want := c.resolveUnit(st.Unit, st.Line)
			if t, ok := rhs.(tTensor); ok {
				if len(t.unit) != 0 && !unitEqual(t.unit, want) {
					c.report(st.Line, "%q declares unit %s but the value has unit %s",
						st.Name, unitString(want), unitString(t.unit))
				}
				env.define(st.Name, tTensor{dims: t.dims, unit: want})
			} else {
				env.define(st.Name, tTensor{dims: []int{}, unit: want})
			}
		} else {
			env.define(st.Name, rhs)
		}
	case *ast.FnDecl:
		env.define(st.Name, tFn{node: st, params: st.Params, ret: st.Ret, retUnit: st.RetUnit, body: st.Body, env: env})
		// Check the body once at definition using the parameter annotations, so
		// mistakes are caught even if the function is never called. Unannotated
		// parameters are unknown, so this only reports definite errors.
		c.checkFnDef(st, env)
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
		// Imported definitions are resolved at runtime. For a namespaced
		// import, bind the alias so field access on it stays unknown rather
		// than "undefined".
		if st.Alias != "" {
			env.define(st.Alias, tUnknown{})
		}
	case *ast.TypeDecl, *ast.UnitDecl:
		// Already registered in the pre-pass.
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
		return tFn{node: ex, params: ex.Params, ret: ex.Ret, retUnit: ex.RetUnit, body: ex.Body, env: env}
	case *ast.Unary:
		return c.inferUnary(ex, env)
	case *ast.Binary:
		return c.inferBinary(ex, env)
	case *ast.Call:
		return c.inferCall(ex, env)
	case *ast.Index:
		return c.inferIndex(ex, env)
	case *ast.Slice:
		return c.inferSlice(ex, env)
	case *ast.RecordLit:
		fields := map[string]Type{}
		for _, f := range ex.Fields {
			fields[f.Name] = c.inferExpr(f.Value, env)
		}
		return tRecord{fields: fields}
	case *ast.Field:
		if rec, ok := c.inferExpr(ex.Target, env).(tRecord); ok {
			if ft, ok := rec.fields[ex.Name]; ok {
				return ft
			}
			c.report(ex.Line, "record has no field %q", ex.Name)
			return tUnknown{}
		}
		return tUnknown{}
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

	lt, lok := l.(tTensor)
	rt, rok := r.(tTensor)

	switch op {
	case "==", "!=", "<", "<=", ">", ">=":
		if lok && rok && !unitEqual(lt.unit, rt.unit) {
			c.report(ex.Line, "comparing incompatible units %s and %s", unitString(lt.unit), unitString(rt.unit))
		}
		return tBool{}
	}

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
		return withUnit(res, unitMul(lt.unit, rt.unit))
	case "^":
		return c.powUnit(lt, ex)
	default: // + - * / %
		res, msg := elementwiseResult(lt, rt)
		if msg != "" {
			c.report(ex.Line, "%s", msg)
			return tUnknown{}
		}
		return withUnit(res, c.arithUnit(op, lt, rt, ex.Line))
	}
}

// withUnit attaches a unit to a tTensor result, leaving other types unchanged.
func withUnit(res Type, u unitMap) Type {
	if t, ok := res.(tTensor); ok {
		t.unit = u
		return t
	}
	return res
}

func (c *checker) arithUnit(op string, lt, rt tTensor, line int) unitMap {
	switch op {
	case "+", "-", "%":
		if !unitEqual(lt.unit, rt.unit) {
			c.report(line, "unit mismatch: %s %s %s", unitString(lt.unit), op, unitString(rt.unit))
		}
		return lt.unit
	case "*":
		return unitMul(lt.unit, rt.unit)
	case "/":
		return unitDiv(lt.unit, rt.unit)
	}
	return nil
}

func (c *checker) powUnit(lt tTensor, ex *ast.Binary) Type {
	if len(lt.unit) == 0 {
		return lt
	}
	k, ok := constInt(ex.Right)
	if !ok {
		c.report(ex.Line, "a quantity with unit %s can only be raised to a constant integer power", unitString(lt.unit))
		return withUnit(lt, nil)
	}
	return withUnit(lt, unitPow(lt.unit, k))
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
			return tTensor{dims: []int{}, unit: v.unit}
		}
		return tTensor{dims: v.dims[1:], unit: v.unit}
	case tList:
		return tUnknown{}
	}
	return tUnknown{}
}

func (c *checker) inferSlice(ex *ast.Slice, env *checkEnv) Type {
	t := c.inferExpr(ex.Target, env)
	if ex.Start != nil {
		c.inferExpr(ex.Start, env)
	}
	if ex.End != nil {
		c.inferExpr(ex.End, env)
	}
	switch v := t.(type) {
	case tTensor:
		if len(v.dims) == 0 {
			c.report(ex.Line, "cannot slice a scalar")
			return tUnknown{}
		}
		first := -1
		s, sok := 0, true
		if ex.Start != nil {
			s, sok = constInt(ex.Start)
		}
		if ex.End != nil {
			if e, eok := constInt(ex.End); eok && sok && e-s >= 0 {
				first = e - s
			}
		} else if sok && v.dims[0] >= 0 && v.dims[0]-s >= 0 {
			first = v.dims[0] - s
		}
		return tTensor{dims: append([]int{first}, v.dims[1:]...), unit: v.unit}
	case tList:
		return tList{}
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
	case tList, tBool, tStr, tUnit, tRecord:
		c.report(ex.Line, "value is not callable")
		return tUnknown{}
	}
	return tUnknown{}
}

// paramType gives a parameter's type from its annotation alone (for checking a
// body at definition). Shape variables and unannotated params are unknown.
func (c *checker) paramType(p ast.Param) Type {
	switch {
	case p.Unit != nil:
		return tTensor{dims: []int{}, unit: unitFromAnno(p.Unit)}
	case p.TypeName != "":
		if t, ok := c.types[p.TypeName]; ok {
			return t
		}
		if c.units[p.TypeName] {
			return tTensor{dims: []int{}, unit: unitMap{p.TypeName: 1}}
		}
		return tUnknown{}
	case p.Shape != nil:
		return tTensor{dims: p.Shape.ConcreteDims()}
	default:
		return tUnknown{}
	}
}

// checkReturnUnit reports if the body's unit disagrees with a declared unit
// return.
func (c *checker) checkReturnUnit(line int, name string, retUnit *ast.UnitAnno, bodyType Type) {
	want := unitFromAnno(retUnit)
	if got, ok := bodyType.(tTensor); ok && !unitEqual(got.unit, want) {
		who := "function"
		if name != "" {
			who = fmt.Sprintf("function %q", name)
		}
		c.report(line, "%s returns unit %s but its signature declares %s",
			who, unitString(got.unit), unitString(want))
	}
}

func (c *checker) checkFnDef(fn *ast.FnDecl, env *checkEnv) {
	if c.stack[fn] {
		return
	}
	scope := newEnv(env)
	for _, p := range fn.Params {
		if p.Unit != nil {
			c.resolveUnit(p.Unit, fn.Line) // report undeclared unit names
		}
		scope.define(p.Name, c.paramType(p))
	}
	if fn.RetUnit != nil {
		c.resolveUnit(fn.RetUnit, fn.Line)
	}
	c.stack[fn] = true
	var bodyType Type
	if blk, ok := fn.Body.(*ast.Block); ok {
		bodyType = c.inferBlock(blk, scope)
	} else {
		bodyType = c.inferExpr(fn.Body, scope)
	}
	delete(c.stack, fn)

	if fn.Ret != nil {
		expected := tTensor{dims: fn.Ret.ConcreteDims()}
		if got, ok := bodyType.(tTensor); ok && fullyKnown(expected) && fullyKnown(got) && !shapeMatch(expected, got) {
			c.report(fn.Line, "function %q returns %s but its signature declares %s",
				fn.Name, dimsString(got), dimsString(expected))
		}
	}
	if fn.RetUnit != nil {
		c.checkReturnUnit(fn.Line, fn.Name, fn.RetUnit, bodyType)
	}
}

func (c *checker) inferUserCall(fn tFn, ex *ast.Call, argTypes []Type) Type {
	if len(fn.params) != len(argTypes) {
		c.report(ex.Line, "function expects %d argument(s), got %d", len(fn.params), len(argTypes))
		return tUnknown{}
	}
	// subst maps shape variables (n, k, ...) to the concrete sizes learned from
	// the arguments, so a variable used in several places must agree.
	subst := map[string]int{}
	scope := newEnv(fn.env)
	for i, p := range fn.params {
		switch {
		case p.Unit != nil:
			want := unitFromAnno(p.Unit)
			c.checkArgUnit(ex.Line, i, p.Name, want, argTypes[i])
			scope.define(p.Name, tTensor{dims: []int{}, unit: want})
		case p.TypeName != "":
			if expected, ok := c.types[p.TypeName]; ok {
				c.checkRecordArg(ex.Line, i, p.Name, expected, argTypes[i])
				scope.define(p.Name, expected) // field access in the body is typed
			} else if c.units[p.TypeName] {
				want := unitMap{p.TypeName: 1}
				c.checkArgUnit(ex.Line, i, p.Name, want, argTypes[i])
				scope.define(p.Name, tTensor{dims: []int{}, unit: want})
			} else {
				c.report(ex.Line, "unknown type %q on parameter %q", p.TypeName, p.Name)
				scope.define(p.Name, argTypes[i])
			}
		case p.Shape != nil:
			if got, ok := argTypes[i].(tTensor); ok {
				c.unify(ex.Line, i, p.Name, p.Shape.Dims, got.dims, subst)
				scope.define(p.Name, got) // use the concrete arg shape in the body
			} else {
				scope.define(p.Name, tTensor{dims: p.Shape.ConcreteDims()})
			}
		default:
			scope.define(p.Name, argTypes[i])
		}
	}
	// Guard against infinite recursion during inference.
	if c.stack[fn.node] {
		if fn.ret != nil {
			return tTensor{dims: substitute(fn.ret.Dims, subst)}
		}
		if fn.retUnit != nil {
			return tTensor{dims: []int{}, unit: unitFromAnno(fn.retUnit)}
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
		expected := tTensor{dims: substitute(fn.ret.Dims, subst)}
		if got, ok := bodyType.(tTensor); ok {
			if fullyKnown(expected) && fullyKnown(got) && !shapeMatch(expected, got) {
				c.report(ex.Line, "function returns %s but its signature declares %s",
					dimsString(got), dimsString(expected))
			}
			expected.unit = got.unit // carry the body's unit through a shape return
		}
		return expected
	}
	if fn.retUnit != nil {
		c.checkReturnUnit(ex.Line, "", fn.retUnit, bodyType)
		return tTensor{dims: []int{}, unit: unitFromAnno(fn.retUnit)}
	}
	return bodyType
}

// checkArgUnit reports if a scalar argument's unit disagrees with a unit
// parameter's declaration.
func (c *checker) checkArgUnit(line, argIdx int, name string, want unitMap, arg Type) {
	// A dimensionless value (a bare literal, or a computed quantity that carries
	// no unit) is adopted into the parameter's unit; only a value that already
	// carries a conflicting unit is an error.
	if got, ok := arg.(tTensor); ok && len(got.unit) != 0 && !unitEqual(got.unit, want) {
		c.report(line, "argument %d (%q) has unit %s but the signature expects %s",
			argIdx+1, name, unitString(got.unit), unitString(want))
	}
}

// checkRecordArg verifies a record argument against a declared record type:
// every declared field must be present with a matching shape.
func (c *checker) checkRecordArg(line, argIdx int, name string, expected tRecord, arg Type) {
	rec, ok := arg.(tRecord)
	if !ok {
		if _, isUnknown := arg.(tUnknown); !isUnknown {
			c.report(line, "argument %d (%q) should be a record", argIdx+1, name)
		}
		return
	}
	for field, ft := range expected.fields {
		got, present := rec.fields[field]
		if !present {
			c.report(line, "argument %d (%q) is missing field %q", argIdx+1, name, field)
			continue
		}
		want, wok := ft.(tTensor)
		gt, gok := got.(tTensor)
		if wok && gok && fullyKnown(want) && fullyKnown(gt) && !shapeMatch(want, gt) {
			c.report(line, "argument %d (%q) field %q is %s but the type declares %s",
				argIdx+1, name, field, dimsString(gt), dimsString(want))
		}
	}
}

// unify checks an argument's shape against a parameter's annotation, recording
// shape variables in subst and reporting any definite mismatch.
func (c *checker) unify(line, argIdx int, name string, pattern []ast.Dim, actual []int, subst map[string]int) {
	if len(pattern) != len(actual) {
		// Only a definite mismatch when the argument's rank is fully known,
		// which it is here (actual comes from a tTensor).
		c.report(line, "argument %d (%q) has rank %d but the signature expects rank %d",
			argIdx+1, name, len(actual), len(pattern))
		return
	}
	for i, pd := range pattern {
		ad := actual[i]
		switch {
		case pd.IsConcrete():
			if ad >= 0 && ad != pd.Size {
				c.report(line, "argument %d (%q) axis %d is %d but the signature expects %d",
					argIdx+1, name, i, ad, pd.Size)
			}
		case pd.Var != "":
			if ad >= 0 {
				if prev, ok := subst[pd.Var]; ok && prev != ad {
					c.report(line, "shape variable %q is %d elsewhere but %d in argument %d",
						pd.Var, prev, ad, argIdx+1)
				} else {
					subst[pd.Var] = ad
				}
			}
		}
	}
}

// substitute resolves an annotation's dims using known shape variables,
// leaving -1 for anything still unknown.
func substitute(dims []ast.Dim, subst map[string]int) []int {
	out := make([]int, len(dims))
	for i, d := range dims {
		switch {
		case d.IsConcrete():
			out[i] = d.Size
		case d.Var != "":
			if v, ok := subst[d.Var]; ok {
				out[i] = v
			} else {
				out[i] = -1
			}
		default:
			out[i] = -1
		}
	}
	return out
}

func (c *checker) inferBuiltinCall(name string, ex *ast.Call, argTypes []Type) Type {
	switch name {
	case "relu", "abs", "softmax":
		// Shape and unit preserved.
		if len(argTypes) >= 1 {
			if t, ok := argTypes[0].(tTensor); ok {
				return t
			}
		}
		return tUnknown{}
	case "exp", "log", "sin", "cos", "tanh", "sigmoid":
		// Transcendental functions need a dimensionless argument.
		if len(argTypes) >= 1 {
			if t, ok := argTypes[0].(tTensor); ok {
				if len(t.unit) != 0 {
					c.report(ex.Line, "%s expects a dimensionless argument but got unit %s", name, unitString(t.unit))
				}
				return tTensor{dims: t.dims} // result is dimensionless
			}
		}
		return tUnknown{}
	case "sqrt":
		if len(argTypes) >= 1 {
			if t, ok := argTypes[0].(tTensor); ok {
				u, ok := unitSqrt(t.unit)
				if !ok {
					c.report(ex.Line, "sqrt of unit %s is not a whole unit", unitString(t.unit))
				}
				return tTensor{dims: t.dims, unit: u}
			}
		}
		return tUnknown{}
	case "square":
		if len(argTypes) >= 1 {
			if t, ok := argTypes[0].(tTensor); ok {
				return tTensor{dims: t.dims, unit: unitPow(t.unit, 2)}
			}
		}
		return tUnknown{}
	case "int":
		return tTensor{dims: []int{}}
	case "item", "len":
		return scalar()
	case "sum", "mean", "max", "min", "prod", "median":
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
	case "reshape", "broadcast_to":
		if len(ex.Args) >= 2 {
			if dims, ok := constShape(ex.Args[1:]); ok {
				return tTensor{dims: dims}
			}
		}
		return tUnknown{}
	case "einsum":
		return c.inferEinsum(ex, argTypes)
	case "concat", "fold":
		return tUnknown{}
	case "append", "enumerate", "columns", "split":
		return tList{}
	case "write_frame":
		return tUnit{}
	case "scalar":
		return scalar()
	case "pow":
		if len(argTypes) >= 1 {
			if t, ok := argTypes[0].(tTensor); ok {
				if len(t.unit) == 0 {
					return t
				}
				if len(ex.Args) >= 2 {
					if k, ok := constInt(ex.Args[1]); ok {
						return tTensor{dims: t.dims, unit: unitPow(t.unit, k)}
					}
				}
				c.report(ex.Line, "pow of a quantity with unit %s needs a constant integer exponent", unitString(t.unit))
				return tTensor{dims: t.dims}
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
				return withUnit(res, unitMul(a.unit, b.unit))
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
	case "range", "list", "map", "zip", "permutation":
		return tList{}
	case "print":
		return tUnit{}
	case "save":
		return tUnit{}
	case "load":
		// The loaded value's type depends on the file, so leave it unknown.
		return tUnknown{}
	case "str":
		return tStr{}
	case "grad", "grads", "value_and_grad", "jacobian", "hessian", "tensor":
		// These return values whose shape depends on runtime data; treat the
		// result as unknown so downstream code is not falsely flagged.
		return tUnknown{}
	case "cumsum", "cumprod", "cummax", "cummin", "floor", "ceil", "round":
		// A cumulative scan or elementwise rounding preserves the input's shape.
		if t, ok := argTypes[0].(tTensor); ok {
			return tTensor{dims: t.dims}
		}
		return tUnknown{}
	case "gather":
		// Selecting rows keeps the trailing dims but the row count is dynamic.
		if t, ok := argTypes[0].(tTensor); ok && len(t.dims) >= 1 {
			dims := make([]int, len(t.dims))
			dims[0] = -1
			copy(dims[1:], t.dims[1:])
			return tTensor{dims: dims}
		}
		return tUnknown{}
	case "conv2d":
		return convResult(argTypes)
	case "maxpool2d":
		// [C, H, W] -> [C, H/k, W/k]; only the channel count is statically known.
		if t, ok := argTypes[0].(tTensor); ok && len(t.dims) == 3 {
			return tTensor{dims: []int{t.dims[0], -1, -1}}
		}
		return tTensor{dims: []int{-1, -1, -1}}
	case "gbm_fit":
		// An opaque model value.
		return tUnknown{}
	case "gbm_predict":
		// One score per row of the feature matrix (the second argument).
		if len(argTypes) == 2 {
			if t, ok := argTypes[1].(tTensor); ok && len(t.dims) >= 1 {
				return tTensor{dims: []int{t.dims[0]}}
			}
		}
		return tTensor{dims: []int{-1}}
	}
	return tUnknown{}
}

// inferEinsum validates a literal einsum spec and, when the input shapes are
// known, resolves the output shape.
func (c *checker) inferEinsum(ex *ast.Call, argTypes []Type) Type {
	if len(ex.Args) < 2 {
		return tUnknown{}
	}
	lit, ok := ex.Args[0].(*ast.StringLit)
	if !ok {
		return tUnknown{}
	}
	inSubs, outSub, err := tensor.ParseEinsum(lit.Value, len(ex.Args)-1)
	if err != nil {
		c.report(ex.Line, "%s", err.Error())
		return tUnknown{}
	}
	dims := make([][]int, len(inSubs))
	for i := range inSubs {
		t, ok := argTypes[i+1].(tTensor)
		if !ok {
			// Spec is valid but an operand's shape is unknown: known rank only.
			od := make([]int, len(outSub))
			for j := range od {
				od[j] = -1
			}
			return tTensor{dims: od}
		}
		dims[i] = t.dims
	}
	out, err := tensor.EinsumOutputDims(inSubs, outSub, dims)
	if err != nil {
		c.report(ex.Line, "%s", err.Error())
		return tUnknown{}
	}
	return tTensor{dims: out}
}

// convResult infers the output shape of conv2d: input [Cin, H, W] and weight
// [Cout, Cin, KH, KW] give [Cout, H-KH+1, W-KW+1], with any unknown dimension
// left as -1.
func convResult(argTypes []Type) Type {
	dims := []int{-1, -1, -1}
	if len(argTypes) != 2 {
		return tTensor{dims: dims}
	}
	in, okIn := argTypes[0].(tTensor)
	w, okW := argTypes[1].(tTensor)
	if okW && len(w.dims) == 4 {
		dims[0] = w.dims[0] // Cout
		if okIn && len(in.dims) == 3 {
			if in.dims[1] >= 0 && w.dims[2] >= 0 {
				dims[1] = in.dims[1] - w.dims[2] + 1
			}
			if in.dims[2] >= 0 && w.dims[3] >= 0 {
				dims[2] = in.dims[2] - w.dims[3] + 1
			}
		}
	}
	return tTensor{dims: dims}
}

// reduceResult handles sum/mean/max/min: no axis reduces to a scalar; a
// constant axis over a known shape removes that dimension.
func (c *checker) reduceResult(ex *ast.Call, argTypes []Type) Type {
	var u unitMap // reductions preserve the input's unit
	if t, ok := argTypes[0].(tTensor); ok {
		u = t.unit
	}
	if len(argTypes) == 1 {
		return tTensor{dims: []int{}, unit: u}
	}
	if len(argTypes) == 2 {
		if t, ok := argTypes[0].(tTensor); ok && len(t.dims) > 0 {
			if ax, ok := constInt(ex.Args[1]); ok {
				if ax < 0 {
					ax += len(t.dims)
				}
				if ax >= 0 && ax < len(t.dims) {
					return tTensor{dims: removeDim(t.dims, ax), unit: u}
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
	case tBool, tStr, tUnit, tList, tRecord, tFn, tBuiltin:
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
	"cos": true, "tanh": true, "sigmoid": true, "sqrt": true, "sum": true, "prod": true, "median": true,
	"mean": true, "abs": true, "pow": true, "matmul": true, "dot": true,
	"grad": true, "grads": true, "value_and_grad": true, "map": true, "zip": true,
	"tensor": true, "scalar": true, "zeros": true, "ones": true, "fill": true,
	"randn": true, "rand": true, "eye": true, "transpose": true, "shape": true,
	"len": true, "item": true, "range": true, "list": true, "str": true,
	"square": true, "maximum": true, "minimum": true, "greater": true,
	"less": true, "greater_equal": true, "less_equal": true, "equal": true,
	"where": true, "clip": true, "max": true, "min": true, "argmax": true,
	"softmax": true, "logsumexp": true, "reshape": true, "broadcast_to": true, "concat": true,
	"fold": true, "append": true, "enumerate": true, "read_csv": true,
	"einsum": true, "map_leaves": true, "zip_leaves": true, "seed": true,
	"read_frame": true, "write_frame": true, "columns": true, "field": true,
	"with_field": true, "gbm_fit": true, "gbm_predict": true,
	"cumsum": true, "cumprod": true, "cummax": true, "cummin": true,
	"conv2d": true, "maxpool2d": true, "save": true, "load": true,
	"gather": true, "permutation": true, "int": true,
	"floor": true, "ceil": true, "round": true, "jacobian": true, "hessian": true,
}
