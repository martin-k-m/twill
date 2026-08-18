package interp_test

import (
	"strings"
	"testing"
)

// Enum values compare structurally, like every other value: two Some(1) built
// separately are equal. They used to fall through `==` to Go pointer identity,
// so `Ok(3) == Ok(3)` was false and no caller expected that.
func TestVariantsCompareStructurally(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
enum Shape { Circle(F64), Sq(I64), Empty }
print(Some(1) == Some(1), Ok("a") == Ok("a"), Some(1) == Some(2), Some(1) == None)
print(Circle(2.0) == Circle(2.0), Circle(2.0) == Sq(2), Empty == Empty)
print(Some([1.0, 2.0]) == Some([1.0, 2.0]))
let d1: Dict[Str, I64] = {}
let d2: Dict[Str, I64] = {}
dict_set(d1, "a", 1)
dict_set(d2, "a", 1)
print(d1 == d2)
`)
	expectLines(t, out, "true true false false", "true false true", "true", "true")
}

// An enum payload declared I64 is stored as an I64, so a match binding on it
// carries the exact integer.
func TestEnumPayloadTypeIsHonoured(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
enum Tok { Num(I64), Word(Str) }
let t: Tok = Num(9007199254740993)
match t { Num(n) => print(n, n / 2), Word(_) => print("word") }
`)
	expectLines(t, out, "9007199254740993 4503599627370496")
}

// `?` at the top of a file has no function to return the failure from. It
// used to end the program quietly with status 0; now it is an error naming the
// value.
func TestTryOutsideAFunctionIsAnError(t *testing.T) {
	_, err := runSrcErr(t, "mode systems\nfn f() -> Res[I64, Str] { Err(\"boom\") }\nlet v = f()?\nprint(\"unreached\")\n")
	if err == nil || !strings.Contains(err.Error(), "`?` outside a function") || !strings.Contains(err.Error(), "Err(boom)") {
		t.Fatalf("err = %v, want the top-level ? error naming Err(boom)", err)
	}
}
