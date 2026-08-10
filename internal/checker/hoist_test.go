package checker_test

import "testing"

// A top-level `let` constant may be referenced above its definition line, the
// same forward reference already granted to functions and enum variants. The
// compiler's own DTYPE_MAKERS table is used this way.
func TestTopLevelLetForwardReference(t *testing.T) {
	wantNone(t, "mode systems\nfn f() -> Str = NAMES\nlet NAMES: Str = \"x y\"")
}

func TestTopLevelLetForwardReferenceNumericMode(t *testing.T) {
	// The hoist is not systems-only: a name that exists further down is not a typo
	// in either dialect.
	wantNone(t, "fn f() = g\nlet g = 3.0")
}

// A capitalized constructor name borrowed from an aliased import is not reported
// as unknown: enum variants register program-wide at run time, so a single-file
// checker cannot prove one absent. This is what lets parse.tw construct ast.tw's
// SFn/EBlock variants unqualified.
func TestCrossModuleVariantTolerated(t *testing.T) {
	wantNone(t, "mode systems\nimport \"ast.tw\" as ast\nfn f() = SFn(ast.mk())")
}

// The tolerance is narrow: a lowercase unknown is a value or function typo and is
// still reported even with an aliased import present.
func TestCrossModuleLowercaseStillReported(t *testing.T) {
	wantOne(t, "mode systems\nimport \"ast.tw\" as ast\nfn f() = nope(1.0)", "unknown name")
}

// And without an aliased import there is no module a variant could come from, so
// even a capitalized unknown is reported.
func TestCapitalizedUnknownWithoutImportReported(t *testing.T) {
	wantOne(t, "mode systems\nfn f() = Nope(1.0)", "unknown name")
}

// `+` on two strings type-checks to a string; a string plus a number is a type
// error, since str() is how a number is meant to join a string.
func TestStringConcatChecks(t *testing.T) {
	wantNone(t, "mode systems\nlet s: Str = \"a\" + \"b\"")
	wantOne(t, "mode systems\nlet s = \"a\" + 3.0", "joins two strings or two numbers")
}
