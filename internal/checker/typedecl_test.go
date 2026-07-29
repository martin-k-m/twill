package checker_test

import "testing"

const modelType = "type Model = { w: [3, 2], b: [3] }\n"

func TestDeclaredTypeValid(t *testing.T) {
	src := modelType + `
		fn predict(m: Model, x: [2]) -> [3] { m.w @ x + m.b }
		let good = { w: [[1.0,2.0],[3.0,4.0],[5.0,6.0]], b: [0.0, 0.0, 0.0] }
		let r = predict(good, [1.0, 1.0])`
	wantNone(t, src)
}

func TestDeclaredTypeWrongFieldShape(t *testing.T) {
	src := modelType + `
		fn predict(m: Model, x: [2]) -> [3] { m.w @ x + m.b }
		let wrong = { w: [[1.0, 2.0]], b: [0.0, 0.0, 0.0] }
		let r = predict(wrong, [1.0, 1.0])`
	wantOne(t, src, "declares")
}

func TestDeclaredTypeMissingField(t *testing.T) {
	src := modelType + `
		fn predict(m: Model, x: [2]) -> [3] { m.w @ x + m.b }
		let partial = { w: [[1.0,2.0],[3.0,4.0],[5.0,6.0]] }
		let r = predict(partial, [1.0, 1.0])`
	wantOne(t, src, "missing field")
}

func TestFieldTypoCaught(t *testing.T) {
	// The body is checked when the function is called; m has type Model, so the
	// field typo `m.wong` is a mistake.
	src := modelType + `
		fn use(m: Model) = sum(m.wong)
		let r = use({ w: [[1.0,2.0],[3.0,4.0],[5.0,6.0]], b: [0.0, 0.0, 0.0] })`
	wantOne(t, src, "no field")
}

func TestUnknownTypeName(t *testing.T) {
	wantOne(t, "fn f(m: Nope) = m\nlet r = f({ a: 1.0 })", "unknown type")
}

func TestFieldTypoOnLiteralCaught(t *testing.T) {
	wantOne(t, "let p = { a: 1.0 }\nlet x = p.b", "no field")
}
