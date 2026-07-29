package checker_test

// Reuses diagnostics/wantOne/wantNone from checker_test.go.

import "testing"

func TestBroadcastingIsAccepted(t *testing.T) {
	// Matrix + row vector is valid broadcasting and must not be flagged.
	wantNone(t, "let m = [[1.0, 2.0], [3.0, 4.0]] + [10.0, 20.0]")
	// Matrix * column vector.
	wantNone(t, "let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] * [[10.0], [100.0]]")
}

func TestNonBroadcastableStillCaught(t *testing.T) {
	// [2] and [3] cannot broadcast.
	wantOne(t, "let z = [1.0, 2.0] + [1.0, 2.0, 3.0]", "broadcast")
	// A matrix whose trailing dim (3) doesn't match a length-2 vector.
	wantOne(t, "let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] + [1.0, 2.0]", "broadcast")
}

func TestReduceAxisShape(t *testing.T) {
	// sum over axis 0 of a [2,3] gives [3]; adding a [2] vector must fail.
	wantOne(t, "let s = sum([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 0) + [1.0, 2.0]", "broadcast")
	// ...but adding a [3] vector is fine.
	wantNone(t, "let s = sum([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 0) + [1.0, 2.0, 3.0]")
}

func TestReshapeShapeTracked(t *testing.T) {
	// reshape to [2,3]; matmul with a [2] vector should fail (inner 3 != 2).
	wantOne(t, "let r = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)\nlet y = r @ [1.0, 2.0]", "inner")
}
