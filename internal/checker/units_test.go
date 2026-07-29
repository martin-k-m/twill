package checker_test

import "testing"

const units = "unit USD\nunit share\n"

func TestUnitsValid(t *testing.T) {
	// price (USD/share) * quantity (share) = USD, matching the declared return.
	src := units + `
		fn notional(px: USD/share, qty: share) -> USD { px * qty }
		let n = notional(150.0, 100.0)`
	wantNone(t, src)
}

func TestUnitAddMismatch(t *testing.T) {
	src := units + `
		fn bad(px: USD/share, qty: share) -> USD { px + qty }
		let r = bad(150.0, 100.0)`
	wantOne(t, src, "unit mismatch")
}

func TestTranscendentalNeedsDimensionless(t *testing.T) {
	src := units + `
		fn f(px: USD) -> USD { log(px) }
		let r = f(100.0)`
	wantOne(t, src, "dimensionless")
}

func TestReturnUnitMismatch(t *testing.T) {
	src := units + `
		fn wrong(px: USD/share, qty: share) -> share { px * qty }
		let r = wrong(1.0, 2.0)`
	wantOne(t, src, "declares")
}

func TestComparisonUnitMismatch(t *testing.T) {
	src := units + `
		fn f(a: USD, b: share) { if a > b { print("x") } }
		let r = f(1.0, 2.0)`
	wantOne(t, src, "incompatible units")
}

func TestUnitDivisionCancels(t *testing.T) {
	// A rate (USD/year) times a duration (year) is USD; sqrt of USD^2 is USD.
	src := `
		unit USD
		unit year
		fn interest(rate: USD/year, t: year) -> USD { rate * t }
		fn rms(a: USD, b: USD) -> USD { sqrt(a * a + b * b) }
		let x = interest(5.0, 2.0)
		let y = rms(3.0, 4.0)`
	wantNone(t, src)
}

func TestUnitsDontAffectPlainCode(t *testing.T) {
	// Code without unit annotations is dimensionless and unaffected.
	wantNone(t, "fn f(x) = x + x\nlet r = f([1.0, 2.0])")
}
