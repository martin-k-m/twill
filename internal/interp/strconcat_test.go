package interp_test

import (
	"testing"

	"github.com/twill-lang/twill/internal/interp"
)

// `+` concatenates two strings. This is what lets the terminal and CLI code
// build output with `a + b` instead of nesting bytes.concat.
func TestStringConcat(t *testing.T) {
	if got := runOut(t, "print(\"foo\" + \"bar\")"); got != "foobar" {
		t.Fatalf("string concat = %q, want \"foobar\"", got)
	}
}

// A string added to a number is still a mistake: str() is how a number joins a
// string, so the loose case stays an error rather than a silent coercion.
func TestStringPlusNumberIsError(t *testing.T) {
	var failed bool
	func() {
		defer func() {
			if recover() != nil {
				failed = true
			}
		}()
		ip := interp.New(func(string) {})
		if _, err := ip.Run("print(\"n=\" + 3.0)"); err != nil {
			failed = true
		}
	}()
	if !failed {
		t.Fatal("expected string + number to fail")
	}
}

// arr is a list literal as a call; chr is a one-byte string; slice is a byte
// substring clamped to the string.
func TestArrChrSlice(t *testing.T) {
	if got := runOut(t, "print(len(arr(1.0, 2.0, 3.0)))"); got != "3" {
		t.Fatalf("len(arr(...)) = %q, want \"3\"", got)
	}
	if got := runOut(t, "print(chr(65) + chr(66))"); got != "AB" {
		t.Fatalf("chr concat = %q, want \"AB\"", got)
	}
	if got := runOut(t, "print(slice(\"hello\", 1, 4))"); got != "ell" {
		t.Fatalf("slice = %q, want \"ell\"", got)
	}
	// slice clamps rather than panicking on an out-of-range end.
	if got := runOut(t, "print(slice(\"hi\", 0, 99))"); got != "hi" {
		t.Fatalf("slice clamp = %q, want \"hi\"", got)
	}
}

// arr_clear empties a list in place, so a shared reference sees the change.
func TestArrClear(t *testing.T) {
	got := runOut(t, "let a = arr(1.0, 2.0)\narr_clear(a)\nprint(len(a))")
	if got != "0" {
		t.Fatalf("len after arr_clear = %q, want \"0\"", got)
	}
}
