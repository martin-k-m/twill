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
