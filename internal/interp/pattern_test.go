package interp_test

import "testing"

// The pattern language (NEEDS-3, 1.7): patterns nest, match literals, and
// carry guards. These pin what each of those does at run time, which is where
// the checker's exhaustiveness reasoning has to be true rather than plausible.

// A nested pattern reaches into the payload, and the arms are tried in order,
// so a narrower one written first wins.
func TestNestedPatternMatchesThroughThePayload(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
fn f(o: Opt[Res[I64, Str]]) -> Str {
  match o {
    Some(Ok(3)) => "three",
    Some(Ok(n)) => "ok",
    Some(Err(e)) => e,
    None => "none",
  }
}
print(f(Some(Ok(3))), f(Some(Ok(4))), f(Some(Err("boom"))), f(None))
`)
	expectLines(t, out, "three ok boom none")
}

// A guard decides an arm after its pattern has matched and its bindings are in
// scope, and a false guard falls through to the arms below rather than failing
// the match.
func TestGuardFallsThroughToTheNextArm(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
fn f(o: Opt[I64]) -> Str {
  match o {
    Some(v) if v > 100 => "big",
    Some(v) if v > 10 => "medium",
    Some(v) => "small",
    None => "none",
  }
}
print(f(Some(500)), f(Some(50)), f(Some(5)), f(None))
`)
	expectLines(t, out, "big medium small none")
}

// A literal pattern matches by the language's own equality, so it works on
// values that are not enums at all -- a match over numbers or strings needs no
// enum to be written around it.
func TestLiteralPatternsMatchPlainValues(t *testing.T) {
	out := runFile(t, t.TempDir(), `fn f(n) {
  match n {
    0 => "zero",
    -1 => "minus one",
    x if x > 100 => "big",
    _ => "other",
  }
}
print(f(0), f(-1), f(500), f(7))
print(match "hi" { "hi" => 1, _ => 2 }, match true { true => 1, false => 2 })
`)
	expectLines(t, out, "zero minus one big other", "1 1")
}

// A lower-case name binds rather than naming a case, which is what makes a
// named catch-all possible. An upper-case one names a case; that rule is what
// keeps `Some(x)` from reading x as a nullary variant.
func TestLowerCaseNameBindsTheWholeValue(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
fn f(o: Opt[I64]) -> Str {
  match o {
    None => "none",
    other => str(other),
  }
}
print(f(None), f(Some(7)))
`)
	expectLines(t, out, "none Some(7)")
}

// A binding that a failed arm made must not survive into the arm that runs.
// Each arm matches into its own scope, which is thrown away on failure.
func TestBindingsDoNotLeakBetweenArms(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
let v: I64 = 99
fn f(o: Opt[Res[I64, Str]]) -> I64 {
  match o {
    Some(Ok(v)) => v,
    _ => v,
  }
}
print(f(Some(Err("x"))), f(Some(Ok(1))))
`)
	expectLines(t, out, "99 1")
}

// A pattern naming a case with no parentheses matches it whatever it carries,
// which is how a payload-carrying case is ignored without naming a binder.
func TestBareVariantPatternIgnoresThePayload(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
enum Tok { Num(I64), Word(Str), End }
fn kind(t: Tok) -> Str {
  match t { Num => "num", Word => "word", End => "end" }
}
print(kind(Num(1)), kind(Word("a")), kind(End))
`)
	expectLines(t, out, "num word end")
}

// User-defined generics (NEEDS-4, 1.7) are erased at run time: the same code
// runs whatever T is, so what these pin is that the syntax reaches the runtime
// unchanged and the values behave as they did without it.
func TestGenericsAreErasedAtRunTime(t *testing.T) {
	out := runFile(t, t.TempDir(), `mode systems
struct Box[T] { value: T }
enum Tree[T] { Leaf(T), Empty }
fn unwrap[T](b: Box[T]) -> T = b.value
fn main() {
  let bi: Box[I64] = Box { value: 3 }
  let bs: Box[Str] = Box { value: "hi" }
  print(unwrap(bi), unwrap(bs))
  let t: Tree[Str] = Leaf("x")
  print(match t { Leaf(v) => v, Empty => "empty" })
}
`)
	expectLines(t, out, "3 hi", "x")
}
