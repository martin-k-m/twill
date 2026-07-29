package checker_test

import "testing"

func TestSliceShapeTracked(t *testing.T) {
	// v[0:2] has shape [2]; adding a [2] vector is fine...
	wantNone(t, "let v = [1.0, 2.0, 3.0, 4.0]\nlet s = v[0:2] + [10.0, 20.0]")
	// ...but adding a [3] vector cannot broadcast.
	wantOne(t, "let v = [1.0, 2.0, 3.0, 4.0]\nlet s = v[0:2] + [10.0, 20.0, 30.0]", "broadcast")
}

func TestSliceMatrixKeepsTrailingDims(t *testing.T) {
	// m[1:3] is [2,2]; @ with a length-2 vector is valid, length-3 is not.
	wantNone(t, "let m = [[1.0,2.0],[3.0,4.0],[5.0,6.0]]\nlet y = m[1:3] @ [1.0, 2.0]")
	wantOne(t, "let m = [[1.0,2.0],[3.0,4.0],[5.0,6.0]]\nlet y = m[1:3] @ [1.0, 2.0, 3.0]", "inner")
}
