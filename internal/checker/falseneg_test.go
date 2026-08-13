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

func TestNegativeConstructorDimCaught(t *testing.T) {
	// These used to reach the runtime and panic in make([]float64, n) with n < 0,
	// surfacing as "makeslice: len out of range" naming no call site.
	wantOne(t, "let a = zeros(-2, 3)", "negative")
	wantOne(t, "let a = ones(3, -1)", "negative")
	wantOne(t, "let a = randn(-4)", "negative")
	wantOne(t, "let a = fill(5.0, -2)", "negative")
	wantOne(t, "let a = eye(-3)", "negative")
	wantOne(t, "let a = linspace(0.0, 1.0, -3)", "negative")
	// Valid constructors stay clean.
	wantNone(t, "let a = zeros(2, 3)")
	wantNone(t, "let a = eye(4)")
	wantNone(t, "let a = linspace(0.0, 1.0, 5)")
}

func TestConvKernelLargerThanInputCaught(t *testing.T) {
	// A kernel bigger than the input produces an empty output; the runtime
	// rejects it, and both spatial pairs are known here.
	wantOne(t, "let x = zeros(3, 8, 8)\nlet y = conv2d(x, zeros(4, 3, 10, 10))", "larger than input")
	// A kernel that fits stays clean.
	wantNone(t, "let x = zeros(3, 8, 8)\nlet y = conv2d(x, zeros(4, 3, 3, 3))")
}

func TestDiffAxisCaught(t *testing.T) {
	// diff's axis was unchecked while flip/roll/scans were not.
	wantOne(t, "let y = diff(zeros(2, 3), 7)", "axis out of range")
	wantNone(t, "let y = diff(zeros(2, 3), 1)")
}

func TestTransposeAxisCountCaught(t *testing.T) {
	// A permutation must name every axis exactly once.
	wantOne(t, "let y = transpose(zeros(2, 3), 0, 1, 2)", "rank-2 tensor")
	wantNone(t, "let y = transpose(zeros(2, 3), 1, 0)")
}

func TestSliceBoundsCaught(t *testing.T) {
	// end past the first dim, and a reversed slice, both reached the runtime.
	wantOne(t, "let a = zeros(3)\nlet b = a[1:9]", "out of range for first dim 3")
	wantOne(t, "let a = zeros(5)\nlet b = a[3:1]", "out of range for first dim 5")
	// Valid slices, open ends, and negative endpoints stay clean.
	wantNone(t, "let a = zeros(5)\nlet b = a[2:4]")
	wantNone(t, "let a = zeros(5)\nlet b = a[1:]")
	wantNone(t, "let a = zeros(5)\nlet b = a[-2:]")
	wantNone(t, "let a = zeros(5)\nlet b = a[:3]")
}
