package checker_test

import "testing"

// The checker enforces the loop-control rules from docs/language-guide.md
// (break/continue): innermost enclosing loop only, and never across a function
// boundary. Without this a stray break reaches the interpreter, which unwinds it
// as an uncaught signal rather than a diagnostic.

func TestBreakOutsideLoopIsReported(t *testing.T) {
	wantOne(t, "let x = 1\nbreak", "break outside a loop")
}

func TestContinueOutsideLoopIsReported(t *testing.T) {
	wantOne(t, "continue", "continue outside a loop")
}

func TestBreakInsideLoopIsFine(t *testing.T) {
	wantNone(t, "for i in range(3) {\n  if i == 1 { break }\n}")
}

func TestContinueInsideWhileIsFine(t *testing.T) {
	wantNone(t, "let i = 0\nwhile i < 3 {\n  i = i + 1\n  if i == 1 { continue }\n}")
}

func TestBreakBindsToInnermostLoopOnly(t *testing.T) {
	// A break in the inner loop is fine; it is the inner loop it leaves.
	wantNone(t, "for i in range(2) {\n  for j in range(2) {\n    break\n  }\n}")
}

func TestBreakCannotCrossAFunctionBoundary(t *testing.T) {
	// A fn written inside a loop is a new scope: the break has no loop to leave.
	wantOne(t, "for i in range(3) {\n  let f = fn() { break }\n  f()\n}", "break outside a loop")
}

func TestLoopDepthUnwindsAfterALoop(t *testing.T) {
	// The depth must return to zero after the loop, so a break below it is caught
	// rather than mistakenly attributed to the loop that has already closed.
	wantOne(t, "for i in range(3) {\n  let x = i\n}\nbreak", "break outside a loop")
}

func TestBreakAfterNestedLoopIsStillCaught(t *testing.T) {
	// Nested increments must decrement symmetrically: two loops closed means
	// depth zero, not depth one.
	wantOne(t, "for i in range(2) {\n  for j in range(2) {\n    let x = j\n  }\n}\ncontinue", "continue outside a loop")
}
