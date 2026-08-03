package interp_test

import (
	"testing"

	"github.com/martin-k-m/raster/internal/interp"
	"github.com/martin-k-m/raster/internal/value"
)

// wantBool runs src and asserts the result is the expected Bool.
func wantBool(t *testing.T, src string, want bool) {
	t.Helper()
	v, _ := run(t, src)
	got, ok := v.(value.Bool)
	if !ok {
		t.Fatalf("%s: expected a bool, got %s", src, value.Format(v))
	}
	if bool(got) != want {
		t.Errorf("%s = %v, want %v", src, bool(got), want)
	}
}

// `==` is structural, so a list equals itself and any list with equal contents.
// It used to answer false for every list, including `a == a`.
func TestListEqualityIsStructural(t *testing.T) {
	cases := map[string]bool{
		"let a = [1.0, \"x\"]\na == a":     true,
		`[1.0, "x"] == [1.0, "x"]`:         true,
		`[1.0, "x"] == [1.0, "y"]`:         false,
		`[1.0, "x"] == [1.0]`:              false,
		`[[1.0], "x"] == [[1.0], "x"]`:     true,
		`[[1.0], "x"] == [[2.0], "x"]`:     false,
		`[1.0, "x"] != [1.0, "x"]`:         false,
		`[1.0, "x"] != [1.0, "y"]`:         true,
		`[{ a: 1.0 }] == [{ a: 1.0 }]`:     true,
		`[{ a: 1.0 }] == [{ a: 2.0 }]`:     false,
		`range(3) == range(3)`:             true,
		`range(3) == range(4)`:             false,
		`[1.0, "x"] == { a: 1.0 }`:         false,
		`[1.0, true] == [1.0, false]`:      false,
		`[1.0, "x"] == "x"`:                false,
		`append([1.0], "x") == [1.0, "x"]`: true,
	}
	for src, want := range cases {
		wantBool(t, src, want)
	}
}

// Records compare field by field, matched by name. This is what makes a
// parameter record comparable to a saved copy of itself.
func TestRecordEqualityIsStructural(t *testing.T) {
	cases := map[string]bool{
		"let m = { w: [1.0, 2.0], b: 0.5 }\nm == m":              true,
		`{ w: [1.0, 2.0], b: 0.5 } == { w: [1.0, 2.0], b: 0.5 }`: true,
		`{ w: [1.0, 2.0], b: 0.5 } == { w: [1.0, 2.0], b: 0.6 }`: false,
		// Field order is presentation, not identity.
		`{ a: 1.0, b: 2.0 } == { b: 2.0, a: 1.0 }`:               true,
		`{ a: 1.0, b: 2.0 } == { a: 1.0 }`:                       false,
		`{ a: 1.0 } == { b: 1.0 }`:                               false,
		`{ a: { b: 1.0 } } == { a: { b: 1.0 } }`:                 true,
		`{ a: { b: 1.0 } } == { a: { b: 2.0 } }`:                 false,
		`{ a: 1.0 } != { a: 1.0 }`:                               false,
		`{ a: 1.0 } != { a: 2.0 }`:                               true,
		`with_field({ a: 1.0 }, "b", 2.0) == { a: 1.0, b: 2.0 }`: true,
	}
	for src, want := range cases {
		wantBool(t, src, want)
	}
}

// A tensor's shape is part of its value, so two tensors holding the same
// numbers in different shapes are not equal.
func TestTensorEqualityComparesShape(t *testing.T) {
	cases := map[string]bool{
		`[1.0, 2.0, 3.0, 4.0] == [1.0, 2.0, 3.0, 4.0]`:                    true,
		`[[1.0, 2.0], [3.0, 4.0]] == [1.0, 2.0, 3.0, 4.0]`:                false,
		`reshape([1.0, 2.0, 3.0, 4.0], 2, 2) == [[1.0, 2.0], [3.0, 4.0]]`: true,
		`[1.0] == 1.0`: false,
		`1.0 == 1.0`:   true,
	}
	for src, want := range cases {
		wantBool(t, src, want)
	}
}

// Functions have no structure to compare, so they compare by identity: the
// same function is equal to itself, two separately written ones are not.
func TestFunctionEqualityIsIdentity(t *testing.T) {
	cases := map[string]bool{
		"let f = fn(x) = x\nf == f":                    true,
		"let f = fn(x) = x\nlet g = f\nf == g":         true,
		"let f = fn(x) = x\nlet g = fn(x) = x\nf == g": false,
		`sum == sum`:  true,
		`sum == mean`: false,
		// Unit is equal to itself; it used to be unequal to everything.
		"let u = if false { 1.0 }\nu == u":   true,
		"let u = if false { 1.0 }\nu == 0.0": false,
	}
	for src, want := range cases {
		wantBool(t, src, want)
	}
}

// Ordering comparisons on structured values are still an error; only equality
// is defined for them.
func TestOrderingOnStructuredValuesErrors(t *testing.T) {
	for _, src := range []string{
		`[1.0, "x"] < [1.0, "y"]`,
		`{ a: 1.0 } < { a: 2.0 }`,
		`"a" < "b"`,
	} {
		ip := interp.New(func(string) {})
		if _, err := ip.Run(src); err == nil {
			t.Errorf("%s: expected an error, got none", src)
		}
	}
}
