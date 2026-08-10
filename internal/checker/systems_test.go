package checker_test

import "testing"

// In a `mode systems` file, a type annotation names a type the bootstrap has no
// definition for (I64, Str, Bool, and module-qualified names like cp.Caps). The
// annotation is advisory there, so an unresolved one is tolerated rather than
// reported. In numeric mode the same annotation is still an error, which is what
// keeps the mode line meaningful rather than a comment.

func TestSystemsModeToleratesUnknownParamAndReturnTypes(t *testing.T) {
	wantNone(t, "mode systems\nfn even(n: I64) -> Bool = n % 2 == 0\nlet r = even(4.0)")
	wantNone(t, "mode systems\nfn label(s: Str) -> Str = s\nlet r = label(1.0)")
}

func TestSystemsModeToleratesQualifiedTypes(t *testing.T) {
	wantNone(t, "mode systems\nfn idc(c: cp.Caps) -> cp.Caps = c\nlet r = idc(1.0)")
}

func TestSystemsModeToleratesTypedLet(t *testing.T) {
	wantNone(t, "mode systems\nlet n: I64 = 3\nlet s: Str = n")
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
// relaxes unknown type names, not the unit algebra.
func TestSystemsModeStillChecksDeclaredUnitReturns(t *testing.T) {
	wantOne(t, "mode systems\nunit USD\nunit share\nfn bad(px: USD/share, qty: share) -> USD { px + qty }\nlet r = bad(1.0, 2.0)", "unit mismatch")
}
