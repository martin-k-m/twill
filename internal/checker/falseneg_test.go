package checker_test

import "testing"

// Shape mistakes that used to reach the runtime untouched, now caught statically.

func TestReshapeNegativeDimCaught(t *testing.T) {
	// twill has no -1 inference, so a negative target dim is a certain error.
	wantOne(t, "let y = reshape(zeros(2, 3), -1, 4)", "-1 dimension inference")
	wantOne(t, "let y = reshape(zeros(2, 3), -1, 3)", "-1 dimension inference")
	// A real element-count mismatch still reported, a valid reshape still clean.
	wantOne(t, "let y = reshape(zeros(2, 3), 5, 5)", "number of elements")
	wantNone(t, "let y = reshape(zeros(2, 3), 3, 2)")
	wantNone(t, "let y = reshape(zeros(2, 3), 6)")
}

func TestGatherIndexRankCaught(t *testing.T) {
	// A rank-2 index cannot select rows.
	wantOne(t, "let y = gather(zeros(3, 4), zeros(2, 2))", "1-D tensor or list of indices")
	// A 1-D index is fine.
	wantNone(t, "let y = gather(zeros(3, 4), zeros(2))")
}

func TestItemSingleElementCaught(t *testing.T) {
	wantOne(t, "let y = item(zeros(3))", "single-element")
	// A scalar or a 1-element tensor is fine; a reduction to a scalar is fine.
	wantNone(t, "let y = item(zeros(1))")
	wantNone(t, "let y = item(sum(zeros(3)))")
}
