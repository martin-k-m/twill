package checker_test

import "testing"

func TestRecordFieldShapeTracked(t *testing.T) {
	// m.w is [1, 3]; @ with a length-2 vector is a definite mismatch.
	wantOne(t, "let m = { w: [[1.0, 2.0, 3.0]] }\nlet y = m.w @ [1.0, 2.0]", "inner")
	// ...but a length-3 vector is fine.
	wantNone(t, "let m = { w: [[1.0, 2.0, 3.0]] }\nlet y = m.w @ [1.0, 2.0, 3.0]")
}

func TestRecordArithmeticIsTypeError(t *testing.T) {
	wantOne(t, "let r = { a: 1.0 }\nlet x = r + 1.0", "numbers/tensors")
}

func TestCallingRecordIsError(t *testing.T) {
	wantOne(t, "let r = { a: 1.0 }\nlet x = r(3.0)", "not callable")
}

func TestNamespacedImportNoFalsePositive(t *testing.T) {
	// The alias is unknown to the checker, so field access stays quiet.
	wantNone(t, "import \"std/nn.ra\" as nn\nlet y = nn.dense([[1.0]], [0.0], [1.0])")
}
