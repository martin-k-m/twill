package lexer

import "testing"

// TestUnterminatedStringEndingInABackslash is the regression test for NEEDS-33.
//
// Source ending in an unterminated string whose last byte is a backslash made
// the string branch consume the backslash and then call advance() for the
// escaped character without checking that one existed, indexing past the end of
// the rune slice and panicking with "index out of range". A panic is the worst
// failure mode a syntax error can have: it names neither the file nor the
// mistake, and a tool that embeds the lexer takes the process down with it.
//
// The bug was found by differential testing against the self-hosted lexer in
// src/lex.tw, which checked and reported "unterminated string" at the opening
// quote. That is the better diagnosis, since the file's problem is the missing
// close quote rather than the backslash, and it is what the Go lexer reports now.
func TestUnterminatedStringEndingInABackslash(t *testing.T) {
	for _, src := range []string{
		`x = "ab\`,
		`"\`,
		`let s = "line\`,
		// A backslash inside an otherwise closed string is untouched by the fix.
		"\"a\\", // just the quote, an 'a' and a backslash
	} {
		toks, _, err := TokenizeWithComments(src)
		if err == nil {
			t.Errorf("%q: expected an unterminated-string error, got tokens %v", src, toks)
			continue
		}
		se, ok := err.(*SyntaxError)
		if !ok {
			t.Errorf("%q: expected a *SyntaxError, got %T", src, err)
			continue
		}
		if se.Msg != "unterminated string" {
			t.Errorf("%q: error is %q, want \"unterminated string\"", src, se.Msg)
		}
	}
}

// A well-formed escape at the end of a closed string still lexes.
func TestEscapesAtTheEndOfAClosedStringStillLex(t *testing.T) {
	toks, _, err := TokenizeWithComments(`let s = "ok\n"`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var got string
	for _, tk := range toks {
		if tk.Kind == STRING {
			got = tk.Value
		}
	}
	if got != "ok\n" {
		t.Errorf("string literal lexed as %q, want %q", got, "ok\n")
	}
}
