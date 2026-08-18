package checker_test

import "testing"

// In a `mode systems` file a type annotation names one of the dialect's types
// (docs/language-guide.md, "Systems-mode types"), and as of 1.6 the checker
// knows what those are: a definite mismatch between a declaration and the value
// put there is reported. Before 1.6 every such annotation was advisory, so
// `let x: I64 = "hello"` passed the check and failed, or quietly did not fail,
// at run time.
//
// The policy is still the shape checker's: report only what is certain. A name
// the checker cannot resolve -- one from another module, an unbound type
// parameter -- judges nothing, and neither does a value whose type is unknown.

func TestSystemsTypeMismatchIsReported(t *testing.T) {
	wantOne(t, "mode systems\nlet x: I64 = \"hello\"", `"x" is declared I64 but the value is Str`)
	wantOne(t, "mode systems\nlet s: Str = 42", `"s" is declared Str but the value is F64`)
	wantOne(t, "mode systems\nlet b: Bool = 1", `"b" is declared Bool but the value is F64`)
	wantOne(t, "mode systems\nfn f(x: Bool) -> I64 { x }", `function "f" returns Bool but its signature declares I64`)
	wantOne(t, "mode systems\nfn g(a: I64) -> Str { a }", `function "g" returns I64 but its signature declares Str`)
	wantOne(t, "mode systems\nlet xs: Arr[I64] = [\"hello\"]", `"xs" is declared Arr[I64] but the value is a list`)
}

// A fraction bound at an I64 truncates, so a written one is always a mistake.
// A whole number is not: `let n: I64 = 3` is how every count is written.
func TestFractionalLiteralAtIntIsReported(t *testing.T) {
	wantOne(t, "mode systems\nlet y: I64 = 2.5", `"y" is declared I64 but the value is the fraction 2.5`)
	wantNone(t, "mode systems\nlet n: I64 = 3\nlet m: F64 = 3\nlet k: I64 = m / 2")
}

// An argument is checked against its parameter's declared type, and a `return`
// against the function's.
func TestArgumentAndReturnTypesAreChecked(t *testing.T) {
	wantOne(t, "mode systems\nfn takes(a: Str) -> Unit { print(a) }\nlet r = takes(3)",
		`argument 1 ("a") is declared Str but the value is F64`)
	wantOne(t, "mode systems\nfn h(n: I64) -> I64 {\n  if n > 0 { return \"neg\" }\n  n\n}",
		`return gives Str but the function declares I64`)
	wantNone(t, "mode systems\nfn takes(a: Str) -> Unit { print(a) }\nlet s: Str = \"x\"\nlet r = takes(s)")
}

// A struct's field types are its declaration's, at construction and at every
// later assignment, and a field it does not declare is a typo.
func TestStructFieldTypesAreChecked(t *testing.T) {
	const decl = "mode systems\nstruct Lexer { src: Str, i: I64 }\n"
	wantOne(t, decl+"let lx: Lexer = Lexer { src: 3, i: 0 }",
		`field "src" of Lexer is declared Str but the value is F64`)
	wantOne(t, decl+"let lx: Lexer = Lexer { src: \"a\", i: 0, nope: 1 }",
		`Lexer has no field "nope"`)
	wantOne(t, decl+"let lx: Lexer = Lexer { src: \"a\", i: 0 }\nlx.i = \"x\"",
		`field "i" of Lexer is declared I64 but the value is Str`)
	wantNone(t, decl+"let lx: Lexer = Lexer { src: \"a\", i: 0 }\nlx.i = lx.i + 1\nlet c: I64 = lx.src[lx.i]")
}

// An enum payload's declared type is checked where the case is constructed,
// and a match binding carries the payload's type into the arm.
func TestEnumPayloadTypesAreChecked(t *testing.T) {
	const decl = "mode systems\nenum Tok { Num(I64), Word(Str), Eof }\n"
	wantOne(t, decl+"let t: Tok = Num(\"x\")", `the payload of Num is declared I64 but the value is Str`)
	wantOne(t, decl+"let t: Tok = Word(\"x\")\nmatch t { Num(n) => print(n), Word(w) => print(w + 1), Eof => print(\"e\") }",
		`operator "+" joins two strings or two numbers, not a mix`)
	wantNone(t, decl+"let t: Tok = Num(3)\nmatch t { Num(n) => print(n + 1), Word(w) => print(w + \"!\"), Eof => print(\"e\") }")
}

// Opt and Res carry their argument types, so a wrongly-typed one is caught
// where it is bound, and a builtin returning one is typed.
func TestOptAndResArgumentsAreChecked(t *testing.T) {
	wantOne(t, "mode systems\nlet o: Opt[I64] = Some(\"s\")", `"o" is declared Opt[I64] but the value is Opt[Str]`)
	wantOne(t, "mode systems\nlet r: Res[I64, Str] = read_file(\"x\")", `"r" is declared Res[I64, Str] but the value is Res[Str, Str]`)
	wantNone(t, "mode systems\nlet r: Res[Str, Str] = read_file(\"x\")\nlet o: Opt[I64] = i64_of_str(\"3\")")
}

// What the checker cannot resolve, it does not judge: a type from another
// module, an unbound type parameter, a value whose type is unknown.
func TestSystemsModeToleratesQualifiedTypes(t *testing.T) {
	wantNone(t, "mode systems\nfn idc(c: cp.Caps) -> cp.Caps = c\nlet x = 1.0\nlet r = idc(x)")
}

func TestUnresolvedTypeParameterJudgesNothing(t *testing.T) {
	wantNone(t, "mode systems\nfn f(b: Dict[Str, V]) -> Unit = print(1.0)\nlet d: Dict[Str, I64] = {}\nlet r = f(d)")
}

// A generic-typed parameter is bound as its declared type, not as whatever the
// argument was, so the body may index it.
func TestGenericParamIsBoundAsDeclared(t *testing.T) {
	wantNone(t, "mode systems\nfn head(xs: Arr[I64]) -> I64 = xs[0]\nlet ys: Arr[I64] = arr_new()\nlet r = head(ys)")
}

func TestNumericModeStillRejectsUnknownLetUnit(t *testing.T) {
	wantOne(t, "let n: I64 = 3", "unknown unit")
}

func TestNumericModeStillRejectsUnknownParamType(t *testing.T) {
	// Same signature, no mode line: the unknown type is an error, exactly as
	// before, so a numeric-mode typo is still caught.
	wantOne(t, "fn idc(c: cp.Caps) = c\nlet r = idc(1.0)", "unknown type")
	wantOne(t, "fn f(m: Nope) = m\nlet r = f(1.0)", "unknown type")
}

// A declared unit return is still checked in a systems-mode file: `mode systems`
// relaxes nothing about the unit algebra.
func TestSystemsModeStillChecksDeclaredUnitReturns(t *testing.T) {
	wantOne(t, "mode systems\nunit USD\nunit share\nfn bad(px: USD/share, qty: share) -> USD { px + qty }\nlet r = bad(1.0, 2.0)", "unit mismatch")
}
