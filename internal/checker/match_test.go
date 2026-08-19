package checker_test

import "testing"

// A match is checked for exhaustiveness against the enum its arms name: every
// variant has an arm, or a `_` stands for the rest (docs/type-system.md,
// "match"). Missing cases are named, so the diagnostic says what to add. A
// duplicate arm, an arm after `_`, and a `_` that covers nothing are reported
// as well, because each is a match that does not say what it appears to say.

const verdict = "mode systems\nenum Verdict { Faster, Slower, Same, Noisy, New }\n"

func TestMatchMissingVariantsAreNamed(t *testing.T) {
	wantOne(t, verdict+"fn f(v: Verdict) -> I64 {\n  match v { Faster => 1, Slower => 2, Same => 3 }\n}",
		"match on Verdict is not exhaustive: missing Noisy, New")
}

func TestMatchWithEveryVariantIsExhaustive(t *testing.T) {
	wantNone(t, verdict+"fn f(v: Verdict) -> I64 {\n  match v { Faster => 1, Slower => 2, Same => 3, Noisy => 4, New => 5 }\n}")
}

func TestMatchWildcardCoversTheRest(t *testing.T) {
	wantNone(t, verdict+"fn f(v: Verdict) -> I64 {\n  match v { Faster => 1, _ => 0 }\n}")
}

func TestMatchWildcardCoveringNothingIsUnreachable(t *testing.T) {
	wantOne(t, verdict+"fn f(v: Verdict) -> I64 {\n  match v { Faster => 1, Slower => 2, Same => 3, Noisy => 4, New => 5, _ => 0 }\n}",
		"unreachable match arm: `_` matches nothing")
}

func TestMatchDuplicateArmIsReported(t *testing.T) {
	wantOne(t, verdict+"fn f(v: Verdict) -> I64 {\n  match v { Faster => 1, Faster => 2, _ => 0 }\n}",
		"duplicate match arm: Faster is already handled")
}

func TestMatchArmAfterWildcardIsUnreachable(t *testing.T) {
	wantOne(t, verdict+"fn f(v: Verdict) -> I64 {\n  match v { _ => 0, Faster => 1 }\n}",
		"unreachable match arm: Faster comes after `_`")
}

func TestMatchOnOptAndResIsChecked(t *testing.T) {
	wantOne(t, "mode systems\nfn k(o: Opt[I64]) -> I64 { match o { Some(x) => x } }",
		"match on Opt is not exhaustive: missing None")
	wantOne(t, "mode systems\nfn k(r: Res[I64, Str]) -> I64 { match r { Ok(x) => x } }",
		"match on Res is not exhaustive: missing Err")
	wantNone(t, "mode systems\nfn k(o: Opt[I64]) -> I64 { match o { Some(x) => x, None => 0 } }")
}

func TestMatchMixingTwoEnumsIsReported(t *testing.T) {
	wantOne(t, "mode systems\nfn m(o: Opt[I64]) -> I64 { match o { Some(x) => x, Ok(y) => y } }",
		"match arm Ok belongs to enum Res, but the earlier arms match Opt")
}

func TestMatchOnAnUnseenEnumIsNotJudged(t *testing.T) {
	// A variant declared in another module: the checker cannot know the enum's
	// other cases, so it says nothing rather than something wrong.
	wantNone(t, "mode systems\nimport \"lib.tw\" as lib\nfn f(v) -> I64 { match v { Left => 1 } }")
}

// `?` returns the failure from the enclosing function, so it needs one, and
// that function's return type has to be a Res or an Opt to return it in.
func TestTryOutsideAFunctionIsReported(t *testing.T) {
	wantOne(t, "mode systems\nfn p() -> Res[I64, Str] { Ok(1) }\nlet top = p()?",
		"`?` outside a function")
}

func TestTryInAFunctionWithTheWrongReturnIsReported(t *testing.T) {
	wantOne(t, "mode systems\nfn p() -> Res[I64, Str] { Ok(1) }\nfn bad() -> I64 { p()? }",
		"`?` in a function that returns I64")
}

func TestTryInResAndOptFunctionsIsFine(t *testing.T) {
	wantNone(t, "mode systems\nfn p() -> Res[I64, Str] { Ok(1) }\nfn good() -> Res[I64, Str] { let v: I64 = p()?\n Ok(v) }\nfn o() -> Opt[I64] { let v: I64 = i64_of_str(\"3\")?\n Some(v) }")
}

// A declared return type is what a call produces, whatever walking the body
// concluded. A block body ending in `return` evaluates to Unit as an
// expression, and taking that as the call's type made `g(n)?` report that `?`
// needs a Res -- on the commonest shape the feature has.
func TestABlockBodiedFunctionsReturnTypeReachesItsCallSites(t *testing.T) {
	wantNone(t, `mode systems
fn g(n: I64) -> Res[I64, Str] { return Ok(n) }
fn s(n: I64) -> Str { return "x" }
fn h(n: I64) -> Res[I64, Str] {
  let v: I64 = g(n)?
  Ok(v)
}
fn k(n: I64) -> Str {
  let c: Str = s(n)
  c
}`)
	// And the declared type is still checked at the call: a Str return does not
	// satisfy an I64 binding.
	wantOne(t, `mode systems
fn s(n: I64) -> Str { return "x" }
fn k(n: I64) -> I64 {
  let c: I64 = s(n)
  c
}`, `"c" is declared I64 but the value is Str`)
}

func TestTryInAnUnannotatedFunctionIsNotJudged(t *testing.T) {
	wantNone(t, "mode systems\nfn p() -> Res[I64, Str] { Ok(1) }\nfn f() { p()? }")
}

// `==` and `!=` are deep equality and are defined on everything. Ordering is
// not: it is numbers and strings. An Opt compared against 0 is the shape that
// mistake takes -- it is what a function that used to return -1 looks like
// after it starts returning an Opt -- so it is worth catching where it is
// written rather than where it runs. Found while migrating spool and selvedge
// off their sentinels: the checker could not be trusted to find the call sites,
// so they had to be grepped for.
func TestOrderingOnAnUnorderableTypeIsReported(t *testing.T) {
	wantOne(t, "mode systems\nfn find() -> Opt[I64] { Some(3) }\nfn f() -> Bool { find() < 0 }",
		`cannot order Opt[I64] with "<"`)
	wantOne(t, "mode systems\nfn f(xs: Arr[I64]) -> Bool { xs > xs }",
		`cannot order Arr[I64] with ">"`)
	wantOne(t, "mode systems\nfn f(a: Bool) -> Bool { a <= a }",
		`cannot order Bool with "<="`)
}

func TestEqualityIsStillDefinedOnEverything(t *testing.T) {
	wantNone(t, "mode systems\nfn find() -> Opt[I64] { Some(3) }\nfn f() -> Bool { find() == None }")
	wantNone(t, "mode systems\nfn f(xs: Arr[I64], ys: Arr[I64]) -> Bool { xs != ys }")
}

func TestOrderingOnNumbersAndStringsIsFine(t *testing.T) {
	wantNone(t, "mode systems\nfn f(a: I64, b: I64) -> Bool { a < b }")
	wantNone(t, "mode systems\nfn f(a: Str, b: Str) -> Bool { a < b }")
	wantNone(t, "mode systems\nfn f(a: F64, b: I64) -> Bool { a >= f64(b) }")
	// A type the checker cannot resolve is not judged on a guess.
	wantNone(t, "mode systems\nfn f(a: cp.Thing, b: cp.Thing) -> Bool { a < b }")
}

// MAX_I64 written as a literal is not a fraction, and it is the commonest
// constant in the subset. The check read fractionality from a round trip
// through int64, which is an out-of-range conversion for anything at or above
// 2^63: it landed on MIN_I64, disagreed with itself, and rejected
// `let mx: I64 = 9223372036854775807`. Shipped in 1.6.0 and found by comparing
// the two checkers, because the interpreter tests never run the checker.
func TestMaxI64IsNotAFraction(t *testing.T) {
	wantNone(t, "mode systems\nlet mx: I64 = 9223372036854775807\n")
	wantNone(t, "mode systems\nlet mn: I64 = -9223372036854775808\n")
	wantNone(t, "mode systems\nlet big: I64 = 9007199254740993\n")
	// An integer written in exponent form is still an integer.
	wantNone(t, "mode systems\nlet k: I64 = 2.5e3\n")
}

func TestAFractionAtAnI64IsStillReported(t *testing.T) {
	wantOne(t, "mode systems\nlet x: I64 = 2.5\n", "the fraction 2.5")
	// A negated fraction reaches the digits under the minus, which the old
	// check did not look through at all.
	wantOne(t, "mode systems\nlet x: I64 = -2.5\n", "the fraction 2.5")
}
