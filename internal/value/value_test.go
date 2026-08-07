package value

import "testing"

// Scopes hold their first few bindings inline and spill to a map after that.
// These pin the behaviour that has to hold across the boundary.
func TestEnvInlineAndSpill(t *testing.T) {
	e := NewEnv(nil)
	names := []string{"a", "b", "c", "d", "e", "f", "g"}
	for i, n := range names {
		e.Define(n, Unit{})
		_ = i
	}
	for _, n := range names {
		if _, ok := e.Get(n); !ok {
			t.Errorf("%q was lost across the inline/map boundary", n)
		}
	}
}

func TestEnvRedefinitionLandsWhereTheNameAlreadyIs(t *testing.T) {
	// If a redefined name were appended instead of overwritten, Get would find
	// the stale copy first and the new value would never be seen.
	e := NewEnv(nil)
	e.Define("x", Unit{})
	first := TheUnit
	e.Define("x", first)
	if v, _ := e.Get("x"); v != Value(first) {
		t.Error("a redefinition did not replace the earlier binding")
	}
}

func TestEnvShadowsRatherThanOverwritingTheParent(t *testing.T) {
	parent := NewEnv(nil)
	parent.Define("x", Unit{})
	child := NewEnv(parent)
	child.Define("x", TheUnit)
	if _, ok := parent.Get("x"); !ok {
		t.Error("defining in a child removed the parent's binding")
	}
}

func TestEnvAssignReachesInlineAndMapBindings(t *testing.T) {
	e := NewEnv(nil)
	for _, n := range []string{"a", "b", "c", "d", "spilled"} {
		e.Define(n, Unit{})
	}
	for _, n := range []string{"a", "spilled"} {
		if !e.Assign(n, TheUnit) {
			t.Errorf("Assign could not reach %q", n)
		}
	}
	if e.Assign("missing", TheUnit) {
		t.Error("Assign invented a binding that was never defined")
	}
}
