package checker

import (
	"fmt"
	"strings"

	"github.com/twill-lang/twill/internal/ast"
)

// This file is the systems-mode half of the type lattice: the types a `mode
// systems` file names in its annotations (docs/language-guide.md, "Systems-mode
// types"), how an annotation's text becomes one, and when a value of one type
// may stand where another is declared.
//
// The policy is the shape checker's: report a mismatch only when both sides are
// known and cannot be the same value, and stay quiet otherwise. That is what
// keeps an unannotated program silent. The strict policy in docs/self-hosting.md
// section 1.3, where an unknown surviving inference is itself an error, is
// NEEDS-49 and is deliberately not this.
//
// One softness is kept on purpose: I64 and F64 stand for each other. An `I64`
// annotation converts a number at run time (docs/language-guide.md, "Integer
// division") and every caller relies on it, so a scalar of either kind is
// accepted where the other is declared. A fractional literal at an I64 is the
// exception, since a written `2.5` bound as an integer is always a mistake.

// tInt is the systems-mode I64 (and Byte).
type tInt struct{}

// tBytes is the growable byte buffer.
type tBytes struct{}

// tArr is Arr[T]. A nil elem is an array whose element type is not known.
type tArr struct{ elem Type }

// tDict is Dict[K, V]; nil key or val when not known.
type tDict struct{ key, val Type }

// tEnum is a value of an enum: Opt[T], Res[T, E], or a user enum by name. args
// are the type arguments, nil when not known.
type tEnum struct {
	name string
	args []Type
}

// tCtor is a payload-carrying variant constructor as a value: calling it
// yields the enum. `Some` before its argument is one.
type tCtor struct {
	enum    string
	variant string
	payload Type // the declared payload type, or nil for a generic one
}

// tFnType is a declared function type, `fn(I64) -> F64`: what a parameter or
// field annotated with one holds. Distinct from tFn, whose body is known.
type tFnType struct {
	params []Type
	ret    Type
}

func (tInt) isType()    {}
func (tBytes) isType()  {}
func (tArr) isType()    {}
func (tDict) isType()   {}
func (tEnum) isType()   {}
func (tCtor) isType()   {}
func (tFnType) isType() {}

// annoText is an annotation as written, whichever slot the parser stored it
// in: a qualified or generic name arrives as text and a bare name as a
// one-factor unit annotation. "" when there is no annotation.
func annoText(typeName string, u *ast.UnitAnno) string {
	if typeName != "" {
		return typeName
	}
	if u != nil && len(u.Factors) == 1 && u.Factors[0].Exp == 1 {
		return u.Factors[0].Name
	}
	return ""
}

// parseType turns an annotation's text into a type. Anything it does not
// recognise -- a name from another module, a type alias it cannot see -- is
// unknown, which judges nothing.
func (c *checker) parseType(text string) Type {
	p := &typeParser{s: strings.TrimSpace(text), c: c}
	t := p.parse()
	if p.err || p.pos != len(p.s) {
		return tUnknown{}
	}
	return t
}

type typeParser struct {
	s   string
	pos int
	c   *checker
	err bool
}

func (p *typeParser) skipSpace() {
	for p.pos < len(p.s) && (p.s[p.pos] == ' ' || p.s[p.pos] == '\t') {
		p.pos++
	}
}

func (p *typeParser) ident() string {
	p.skipSpace()
	start := p.pos
	for p.pos < len(p.s) {
		ch := p.s[p.pos]
		if ch == '_' || ch == '.' || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') {
			p.pos++
			continue
		}
		break
	}
	return p.s[start:p.pos]
}

func (p *typeParser) accept(tok string) bool {
	p.skipSpace()
	if strings.HasPrefix(p.s[p.pos:], tok) {
		p.pos += len(tok)
		return true
	}
	return false
}

func (p *typeParser) parse() Type {
	p.skipSpace()
	if p.accept("fn(") || p.accept("Fn(") {
		var params []Type
		if !p.accept(")") {
			for {
				// A parameter may be named: `fn(a: Str, b: Str) -> Bool`.
				save := p.pos
				name := p.ident()
				if name == "" || !p.accept(":") {
					p.pos = save
				}
				params = append(params, p.parse())
				if p.accept(",") {
					continue
				}
				if !p.accept(")") {
					p.err = true
					return tUnknown{}
				}
				break
			}
		}
		var ret Type = tUnit{}
		if p.accept("->") {
			ret = p.parse()
		}
		return tFnType{params: params, ret: ret}
	}
	name := p.ident()
	if name == "" {
		p.err = true
		return tUnknown{}
	}
	var args []Type
	if p.accept("[") {
		for {
			args = append(args, p.parse())
			if p.accept(",") {
				continue
			}
			if !p.accept("]") {
				p.err = true
				return tUnknown{}
			}
			break
		}
	}
	return p.c.namedType(name, args)
}

// namedType resolves a type name with its arguments.
func (c *checker) namedType(name string, args []Type) Type {
	arg := func(i int) Type {
		if i < len(args) {
			return args[i]
		}
		return nil
	}
	// A declaration in the file wins over a built-in name: the self-hosted
	// tensor engine declares its own `struct Tensor`.
	if _, isEnum := c.enums[name]; isEnum {
		return tEnum{name: name, args: args}
	}
	if rec, isRecord := c.types[name]; isRecord {
		return rec
	}
	switch name {
	case "I64", "Byte":
		return tInt{}
	case "F64":
		return scalar()
	case "Bool":
		return tBool{}
	case "Str":
		return tStr{}
	case "Bytes":
		return tBytes{}
	case "Unit":
		return tUnit{}
	case "Tensor":
		// A tensor of any rank. The shape checker's own annotations say the
		// shape; this name says only that it is one, which is not enough to
		// judge anything against, so it is unknown here.
		return tUnknown{}
	case "Arr", "List":
		return tArr{elem: arg(0)}
	case "Dict":
		return tDict{key: arg(0), val: arg(1)}
	case "Opt", "Res":
		return tEnum{name: name, args: args}
	}
	return tUnknown{}
}

// typeString renders a type the way a program would write it, for a
// diagnostic. A tensor with a shape is its shape; a scalar is F64 in systems
// mode.
func (c *checker) typeString(t Type) string {
	switch v := t.(type) {
	case tInt:
		return "I64"
	case tTensor:
		if isScalar(v) {
			if c.systems {
				return "F64"
			}
			return "a scalar"
		}
		return "a tensor of shape " + dimsString(v)
	case tBool:
		return "Bool"
	case tStr:
		return "Str"
	case tBytes:
		return "Bytes"
	case tUnit:
		return "Unit"
	case tList:
		return "a list"
	case tArr:
		if v.elem == nil {
			return "Arr"
		}
		return "Arr[" + c.typeString(v.elem) + "]"
	case tDict:
		if v.key == nil || v.val == nil {
			return "Dict"
		}
		return "Dict[" + c.typeString(v.key) + ", " + c.typeString(v.val) + "]"
	case tEnum:
		if len(v.args) == 0 {
			return v.name
		}
		parts := make([]string, len(v.args))
		for i, a := range v.args {
			if a == nil {
				parts[i] = "_"
			} else {
				parts[i] = c.typeString(a)
			}
		}
		return v.name + "[" + strings.Join(parts, ", ") + "]"
	case tCtor:
		return "the constructor " + v.variant
	case tRecord:
		if v.name != "" {
			return v.name
		}
		return "a record"
	case tFn, tBuiltin, tFnType:
		return "a function"
	}
	return "an unknown value"
}

// assignable reports whether a value of type got may stand where want is
// declared. It is true whenever either side is not known well enough to say
// otherwise, and false only for a definite mismatch.
func assignable(want, got Type) bool {
	if want == nil || got == nil {
		return true
	}
	if _, unk := want.(tUnknown); unk {
		return true
	}
	if _, unk := got.(tUnknown); unk {
		return true
	}
	// A constructor or a function value stands wherever a function is expected;
	// nothing else about it is checked here.
	switch got.(type) {
	case tCtor, tFn, tBuiltin, tFnType:
		_, wantFn := want.(tFnType)
		return wantFn
	}
	switch w := want.(type) {
	case tInt:
		return isNumberType(got)
	case tTensor:
		if isScalar(w) {
			return isNumberType(got)
		}
		_, ok := got.(tTensor)
		// A tensor annotation is checked by the shape rules, not here.
		return ok || isNumberType(got)
	case tBool:
		_, ok := got.(tBool)
		return ok
	case tStr:
		_, ok := got.(tStr)
		return ok
	case tBytes:
		_, ok := got.(tBytes)
		return ok
	case tUnit:
		return true
	case tArr:
		switch g := got.(type) {
		case tArr:
			return g.elem == nil || w.elem == nil || assignable(w.elem, g.elem)
		case tList:
			for _, e := range g.elems {
				if !assignable(w.elem, e) {
					return false
				}
			}
			return true
		case tTensor:
			// A bracket literal of numbers is a tensor to the parser and an array
			// at an Arr annotation, and iterating a rank-n tensor gives rank-(n-1)
			// tensors, which stand for the inner arrays of a nested one. Both are
			// what the runtime does at the annotation, so neither is a mismatch.
			_ = g
			return true
		}
		return false
	case tDict:
		switch g := got.(type) {
		case tDict:
			return (g.key == nil || w.key == nil || assignable(w.key, g.key)) &&
				(g.val == nil || w.val == nil || assignable(w.val, g.val))
		case tRecord:
			// `{}` is a record to the parser and a dictionary at a Dict annotation.
			return len(g.fields) == 0
		}
		return false
	case tEnum:
		g, ok := got.(tEnum)
		if !ok || g.name != w.name {
			return false
		}
		for i := range w.args {
			if i < len(g.args) && !assignable(w.args[i], g.args[i]) {
				return false
			}
		}
		return true
	case tRecord:
		g, ok := got.(tRecord)
		if !ok {
			return false
		}
		if w.name != "" && g.name != "" {
			return w.name == g.name
		}
		return true
	case tFnType:
		switch got.(type) {
		case tFn, tBuiltin, tFnType, tCtor:
			return true
		}
		return false
	}
	return true
}

// isNumberType reports whether a type is a scalar number of either kind: an
// I64, or a rank-0 tensor (F64 in systems mode). A tensor whose rank is not
// known counts, since it may be a scalar.
func isNumberType(t Type) bool {
	switch v := t.(type) {
	case tInt:
		return true
	case tTensor:
		return isScalar(v) || !fullyKnown(v)
	}
	return false
}

// checkAssignable reports a definite mismatch between a declared type and the
// value put there. `what` names the place: `"x"` for a binding, `field "i" of
// Lexer`, `argument 2 ("k")`.
func (c *checker) checkAssignable(line int, what string, want, got Type) bool {
	if assignable(want, got) {
		return true
	}
	c.report(line, "%s is declared %s but the value is %s", what, c.typeString(want), c.typeString(got))
	return false
}

// fractionalLiteralAtInt is the one I64/F64 mismatch that is reported: a
// fractional literal bound where an I64 is declared, since `let n: I64 = 2.5`
// truncates to a number the author did not write.
func (c *checker) fractionalLiteralAtInt(line int, what string, want Type, e ast.Expr) {
	if _, isInt := want.(tInt); !isInt {
		return
	}
	if lit, ok := e.(*ast.NumberLit); ok && lit.Value != float64(int64(lit.Value)) {
		c.report(line, "%s is declared I64 but the value is the fraction %s", what, lit.Text)
	}
}

// bindingType is the type a `let` or a parameter takes from its annotation:
// the declared type when it is one this checker knows, else the value's own,
// so an advisory name from another module still lets the value carry what it
// was.
func bindingType(declared, got Type) Type {
	if _, unk := declared.(tUnknown); unk {
		return got
	}
	// A declared Arr with no element type takes the value's element type when
	// the value has one; and so on for a Dict.
	switch d := declared.(type) {
	case tArr:
		if g, ok := got.(tArr); ok && d.elem == nil {
			return g
		}
	case tDict:
		if g, ok := got.(tDict); ok && d.key == nil {
			return g
		}
	case tEnum:
		if g, ok := got.(tEnum); ok && len(d.args) == 0 {
			return g
		}
	}
	return declared
}

// structFieldTypes parses each declared struct's field types, once every enum
// and struct name is registered, so a field written `Arr[Tok]` resolves.
func (c *checker) structFieldTypes(prog *ast.Program) {
	for _, s := range prog.Body {
		sd, ok := s.(*ast.StructDecl)
		if !ok {
			continue
		}
		fields := map[string]Type{}
		for _, f := range sd.Fields {
			fields[f.Name] = c.parseType(f.Type)
		}
		c.types[sd.Name] = tRecord{fields: fields, name: sd.Name}
		c.structDecls[sd.Name] = sd
	}
}

// variantPayloadType is the declared payload type of an enum case, or nil.
func (c *checker) variantPayloadType(variant string) Type {
	if t, ok := c.payloads[variant]; ok {
		return t
	}
	return nil
}

// enumValueType is the type of a variant reached as a value: `None` is an
// Opt, `Faster` a Verdict. Nil when the name is not a variant this file
// knows.
func (c *checker) enumValueType(name string) (Type, bool) {
	owner, ok := c.variantOwner[name]
	if !ok || owner == "" {
		return nil, false
	}
	if _, hasPayload := c.payloads[name]; hasPayload {
		return tCtor{enum: owner, variant: name, payload: c.payloads[name]}, true
	}
	return tEnum{name: owner}, true
}

// callCtor is the type of applying a variant constructor to its payload:
// Some(x) is Opt[type of x], Ok(x) a Res whose error type is not yet known,
// and a user variant its enum.
func (c *checker) callCtor(ctor tCtor, line int, args []Type, argExprs []ast.Expr) Type {
	if len(args) != 1 {
		c.report(line, "%s expects 1 argument(s), got %d", ctor.variant, len(args))
		return tEnum{name: ctor.enum}
	}
	switch ctor.enum {
	case "Opt":
		return tEnum{name: "Opt", args: []Type{args[0]}}
	case "Res":
		if ctor.variant == "Ok" {
			return tEnum{name: "Res", args: []Type{args[0], nil}}
		}
		return tEnum{name: "Res", args: []Type{nil, args[0]}}
	}
	if ctor.payload != nil {
		what := fmt.Sprintf("the payload of %s", ctor.variant)
		if c.checkAssignable(line, what, ctor.payload, args[0]) && len(argExprs) == 1 {
			c.fractionalLiteralAtInt(line, what, ctor.payload, argExprs[0])
		}
	}
	return tEnum{name: ctor.enum}
}

// matchBindingType is the type a pattern's binding takes: the subject's
// payload for that case, when the subject's type is known.
func (c *checker) matchBindingType(subject Type, pat ast.MatchPattern) Type {
	en, ok := subject.(tEnum)
	if !ok {
		if t := c.variantPayloadType(pat.Variant); t != nil {
			return t
		}
		return tUnknown{}
	}
	switch en.name {
	case "Opt":
		if len(en.args) > 0 && en.args[0] != nil {
			return en.args[0]
		}
	case "Res":
		i := 0
		if pat.Variant == "Err" {
			i = 1
		}
		if len(en.args) > i && en.args[i] != nil {
			return en.args[i]
		}
	default:
		if t := c.variantPayloadType(pat.Variant); t != nil {
			return t
		}
	}
	return tUnknown{}
}

// systemsBuiltinResult types the systems-mode builtins whose result type is
// fixed, so a value from one of them can be checked where it lands. Only
// builtins whose result the shape cases below do not already type are here.
func systemsBuiltinResult(name string, args []Type) (Type, bool) {
	arg := func(i int) Type {
		if i < len(args) {
			return args[i]
		}
		return nil
	}
	switch name {
	case "i64", "i64_of_f64", "clock_now_ms", "mono_ns", "mtime", "file_size", "buf_len", "buf_get8", "rng_open":
		return tInt{}, true
	case "f64", "f64_of_i64":
		return scalar(), true
	case "chr", "str_quote", "f64_to_str", "bytes_to_str", "read_text_or", "resolve_path", "f64_hex", "num_to_text":
		return tStr{}, true
	case "arr_push", "push":
		if a, ok := arg(0).(tArr); ok {
			return a, true
		}
		return tArr{}, true
	case "arr", "arr_new", "arr_of_tensor":
		return tArr{}, true
	case "slice":
		// slice(x, a, b) is a copy of x's range, so it is whatever x was: a Str
		// slices to a Str and an Arr to an Arr.
		switch a := arg(0).(type) {
		case tStr:
			return tStr{}, true
		case tArr:
			return a, true
		}
		return nil, false
	case "dict_new":
		return tDict{}, true
	case "dict_keys":
		return tArr{elem: tStr{}}, true
	case "dict_get":
		if d, ok := arg(0).(tDict); ok && d.val != nil {
			return tEnum{name: "Opt", args: []Type{d.val}}, true
		}
		return tEnum{name: "Opt", args: []Type{nil}}, true
	case "dict_has", "is_same", "is_tty_stdout", "all_finite", "path_exists", "path_is_dir", "path_is_abs":
		return tBool{}, true
	case "path_join", "path_base", "path_dir", "path_ext", "path_stem", "path_normalize":
		return tStr{}, true
	case "mkdir_all", "remove_file", "remove_dir", "remove_all", "rename":
		return tEnum{name: "Res", args: []Type{tUnit{}, tStr{}}}, true
	case "temp_dir", "cwd":
		return tEnum{name: "Res", args: []Type{tStr{}, tStr{}}}, true
	case "bytes_new":
		return tBytes{}, true
	case "read_file":
		return tEnum{name: "Res", args: []Type{tStr{}, tStr{}}}, true
	case "write_file":
		return tEnum{name: "Res", args: []Type{tUnit{}, tStr{}}}, true
	case "i64_of_str":
		return tEnum{name: "Opt", args: []Type{tInt{}}}, true
	case "str_to_f64":
		return tEnum{name: "Opt", args: []Type{scalar()}}, true
	case "env":
		return tEnum{name: "Opt", args: []Type{tStr{}}}, true
	case "args":
		return tArr{elem: tStr{}}, true
	}
	return nil, false
}
