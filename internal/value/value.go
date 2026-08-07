// Package value defines Raster's runtime values and lexical environments.
package value

import (
	"strconv"
	"strings"

	"github.com/martin-k-m/raster/internal/ast"
	"github.com/martin-k-m/raster/internal/tensor"
)

// Value is any runtime value. Concrete types: *tensor.Tensor, Bool, Str,
// *List, *Closure, *Builtin, Unit.
type Value any

type Bool bool
type Str string
type Unit struct{}

type List struct {
	Items []Value
}

// Record is a struct with named fields. Keys preserves declaration order for
// stable printing.
type Record struct {
	Keys   []string
	Fields map[string]Value
}

func NewRecord() *Record {
	return &Record{Fields: map[string]Value{}}
}

func (r *Record) Set(name string, v Value) {
	if _, ok := r.Fields[name]; !ok {
		r.Keys = append(r.Keys, name)
	}
	r.Fields[name] = v
}

func (r *Record) Get(name string) (Value, bool) {
	v, ok := r.Fields[name]
	return v, ok
}

type Closure struct {
	Params []string
	Body   ast.Expr
	Env    *Env
	Name   string
}

// Builtin is a native function. Fn receives evaluated args and returns a
// value or an error.
type Builtin struct {
	Name     string
	Arity    int // -1 means variadic
	Variadic bool
	Fn       func(args []Value) (Value, error)
}

var TheUnit = Unit{}

// --- environments ----------------------------------------------------------

// envInline is how many bindings a scope holds before it needs a map.
//
// Most scopes hold one or two: a loop variable, a couple of parameters. A map
// for that allocates far more than it stores, and an interpreted loop makes a
// scope per iteration, so the map was 36% of the allocations in a scalar loop.
// Four covers the overwhelming majority; beyond it the map takes over and
// nothing else changes.
const envInline = 4

type Env struct {
	// Inline bindings, searched before the map. A linear scan of at most four
	// strings beats a hash for these sizes and costs no allocation at all.
	names  [envInline]string
	values [envInline]Value
	n      int

	vars   map[string]Value // spillover, and every binding in an ordered scope
	parent *Env
	order  []string // definition order, kept only when ordered is set
	// ordered is off for ordinary scopes: a call frame or loop body defines
	// names on every iteration, and paying for a slice append there would show
	// up in a training loop.
	ordered bool
}

// NewEnv creates a scope. The backing map is not allocated until a variable is
// defined, so scopes that only read outer variables cost nothing extra.
func NewEnv(parent *Env) *Env {
	return &Env{parent: parent}
}

// NewModuleEnv creates a scope that remembers the order its names were first
// defined in. A namespaced import snapshots such a scope into a record, and
// that record's field order has to be declaration order rather than Go map
// order for a program to be reproducible run to run.
func NewModuleEnv(parent *Env) *Env {
	return &Env{parent: parent, ordered: true}
}

func (e *Env) Get(name string) (Value, bool) {
	for env := e; env != nil; env = env.parent {
		for i := 0; i < env.n; i++ {
			if env.names[i] == name {
				return env.values[i], true
			}
		}
		if env.vars != nil {
			if v, ok := env.vars[name]; ok {
				return v, true
			}
		}
	}
	return nil, false
}

// Define binds name in this scope.
func (e *Env) Define(name string, v Value) {
	// An ordered scope keeps every binding in the map, because Locals and
	// LocalNames hand that map and its order to the module snapshot.
	if e.ordered {
		if e.vars == nil {
			e.vars = make(map[string]Value, 4)
		}
		if _, redefined := e.vars[name]; !redefined {
			e.order = append(e.order, name)
		}
		e.vars[name] = v
		return
	}

	// Redefinition has to land where the name already is, or Get would find the
	// stale copy first and the new value would never be seen.
	for i := 0; i < e.n; i++ {
		if e.names[i] == name {
			e.values[i] = v
			return
		}
	}
	if e.vars != nil {
		if _, exists := e.vars[name]; exists {
			e.vars[name] = v
			return
		}
	}

	if e.n < envInline {
		e.names[e.n] = name
		e.values[e.n] = v
		e.n++
		return
	}
	if e.vars == nil {
		e.vars = make(map[string]Value, 4)
	}
	e.vars[name] = v
}

// Assign updates the nearest existing binding, returning false if none exists.
func (e *Env) Assign(name string, v Value) bool {
	for env := e; env != nil; env = env.parent {
		for i := 0; i < env.n; i++ {
			if env.names[i] == name {
				env.values[i] = v
				return true
			}
		}
		if env.vars != nil {
			if _, ok := env.vars[name]; ok {
				env.vars[name] = v
				return true
			}
		}
	}
	return false
}

// Locals returns this scope's own bindings (not parents'). Used to snapshot a
// module's definitions into a namespace record.
//
// Inline bindings are folded in, so a caller sees one scope rather than the two
// halves it happens to be stored in. Only a module scope is snapshotted and
// those keep everything in the map, so this copies nothing in practice.
func (e *Env) Locals() map[string]Value {
	if e.n == 0 {
		return e.vars
	}
	out := make(map[string]Value, e.n+len(e.vars))
	for k, v := range e.vars {
		out[k] = v
	}
	for i := 0; i < e.n; i++ {
		out[e.names[i]] = e.values[i]
	}
	return out
}

// LocalNames returns this scope's own bindings in the order they were first
// defined. It is populated only for a scope made by NewModuleEnv.
func (e *Env) LocalNames() []string { return e.order }

// --- helpers ---------------------------------------------------------------

func Truthy(v Value) bool {
	switch t := v.(type) {
	case *tensor.Tensor:
		if t.IsScalar() {
			return t.Data[0] != 0
		}
		return len(t.Data) > 0
	case Bool:
		return bool(t)
	case Str:
		return len(t) > 0
	case *List:
		return len(t.Items) > 0
	case *Record:
		return len(t.Keys) > 0
	case Unit:
		return false
	default:
		return true
	}
}

// Format renders a value for print and the REPL.
func Format(v Value) string {
	switch t := v.(type) {
	case *tensor.Tensor:
		if t.IsScalar() {
			return FormatNumber(t.Data[0])
		}
		return "tensor(" + formatNested(t.ToNested()) + ", shape=[" + joinInts(t.Shape) + "])"
	case Bool:
		if t {
			return "true"
		}
		return "false"
	case Str:
		return string(t)
	case *List:
		parts := make([]string, len(t.Items))
		for i, it := range t.Items {
			parts[i] = Format(it)
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case *Record:
		parts := make([]string, len(t.Keys))
		for i, k := range t.Keys {
			parts[i] = k + ": " + Format(t.Fields[k])
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case *Closure:
		name := t.Name
		if name == "" {
			name = "anonymous"
		}
		return "<fn " + name + "(" + strings.Join(t.Params, ", ") + ")>"
	case *Builtin:
		return "<builtin " + t.Name + ">"
	case Unit:
		return "()"
	default:
		// Native opaque values (e.g. a fitted model) render via String() if they
		// provide one.
		if s, ok := v.(interface{ String() string }); ok {
			return s.String()
		}
		return "<value>"
	}
}

func formatNested(v any) string {
	if s, ok := v.([]any); ok {
		parts := make([]string, len(s))
		for i, item := range s {
			parts[i] = formatNested(item)
		}
		return "[" + strings.Join(parts, ", ") + "]"
	}
	return FormatNumber(v.(float64))
}

// FormatNumber prints integers without a decimal point and trims noise.
func FormatNumber(n float64) string {
	if n == float64(int64(n)) {
		return strconv.FormatInt(int64(n), 10)
	}
	s := strconv.FormatFloat(n, 'f', 6, 64)
	s = strings.TrimRight(s, "0")
	s = strings.TrimRight(s, ".")
	return s
}

func joinInts(xs []int) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = strconv.Itoa(x)
	}
	return strings.Join(parts, ", ")
}
