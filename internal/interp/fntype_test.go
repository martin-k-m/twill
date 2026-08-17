package interp_test

import (
	"testing"

	"github.com/twill-lang/twill/internal/ast"
	"github.com/twill-lang/twill/internal/parser"
)

// A function type (`fn(A, B) -> C`) is the systems-mode annotation for a
// callback, in a parameter, a return, a `let`, or a struct field. It is advisory
// like every other systems type, kept as text. These pin that it parses in all
// four positions, including the higher-order and qualified-name forms the
// training and inference packages use.

func TestFunctionParamTypeParses(t *testing.T) {
	fn := fnDecl(t, "fn apply(f: fn(F64) -> F64, x: F64) -> F64 = f(x)\n")
	if got := fn.Params[0].TypeName; got != "fn(F64) -> F64" {
		t.Fatalf("param type = %q, want %q", got, "fn(F64) -> F64")
	}
}

func TestFunctionReturnTypeParses(t *testing.T) {
	fn := fnDecl(t, "fn make() -> fn(F64) -> F64 = fn(x) = x\n")
	if got := fn.RetType; got != "fn(F64) -> F64" {
		t.Fatalf("return type = %q, want %q", got, "fn(F64) -> F64")
	}
}

func TestHigherOrderAndQualifiedFunctionTypeParses(t *testing.T) {
	// A callback taking a callback, with a module-qualified result: the shape the
	// loom trainer's step signature has.
	fn := fnDecl(t, "fn f(g: fn(Tree, st.OptState) -> st.StepResult) = g\n")
	if got := fn.Params[0].TypeName; got != "fn(Tree, st.OptState) -> st.StepResult" {
		t.Fatalf("param type = %q, want %q", got, "fn(Tree, st.OptState) -> st.StepResult")
	}
	fn2 := fnDecl(t, "fn h(r: fn(fn(F64) -> F64) -> F64) = r\n")
	if got := fn2.Params[0].TypeName; got != "fn(fn(F64) -> F64) -> F64" {
		t.Fatalf("param type = %q, want %q", got, "fn(fn(F64) -> F64) -> F64")
	}
}

func TestFunctionTypeInStructFieldParses(t *testing.T) {
	prog, err := parser.Parse("mode systems\nstruct Callbacks { step: fn(F64) -> F64 }\n")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	sd, ok := prog.Body[0].(*ast.StructDecl)
	if !ok {
		t.Fatalf("first statement is %T, want *ast.StructDecl", prog.Body[0])
	}
	if got := sd.Fields[0].Type; got != "fn(F64) -> F64" {
		t.Fatalf("field type = %q, want %q", got, "fn(F64) -> F64")
	}
}

func TestFunctionTypeInLetParses(t *testing.T) {
	prog, err := parser.Parse("let g: fn(F64) -> F64 = fn(x) = x\n")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	let, ok := prog.Body[0].(*ast.Let)
	if !ok {
		t.Fatalf("first statement is %T, want *ast.Let", prog.Body[0])
	}
	if got := let.TypeName; got != "fn(F64) -> F64" {
		t.Fatalf("let type = %q, want %q", got, "fn(F64) -> F64")
	}
}

// A function-typed program checks clean (advisory types are unchecked) and runs.
func TestFunctionTypedProgramRunsAndChecks(t *testing.T) {
	src := "fn apply(f: fn(F64) -> F64, x: F64) -> F64 = f(x)\n" +
		"let inc: fn(F64) -> F64 = fn(x) = x + 1.0\n" +
		"print(apply(inc, 4.0))\n"
	_, out := run(t, src)
	if len(out) == 0 || out[len(out)-1] != "5" {
		t.Fatalf("program output = %v, want last line \"5\"", out)
	}
}
