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

func TestToCastTypesClean(t *testing.T) {
	// x.to(dt) is a cast: the dtype name is read from the syntax, not inferred as
	// a variable, and the result keeps x's shape and unit.
	wantNone(t, "let y = tensor([1.0, 2.0]).to(f32)")
	wantNone(t, "let y = zeros(2, 3).to(bf16)")
	wantNone(t, "let y = zeros(2, 3).to(bf16).to(f16)") // chains
	// A non-dtype argument is not a cast, so it falls through and the unknown
	// name is reported as usual.
	wantOne(t, "let y = tensor([1.0]).to(nope)", "unknown name \"nope\"")
}

func TestQuantizeRankCaught(t *testing.T) {
	// Quantisation packs a 2-D weight; any other rank is the runtime error.
	wantOne(t, "let y = quantize(zeros(5))", "expects a 2-D weight, got rank 1")
	wantOne(t, "let y = quantize(zeros(2, 3, 4))", "expects a 2-D weight, got rank 3")
	wantOne(t, "let y = quantize(zeros(5), 4)", "expects a 2-D weight, got rank 1")
	// A 2-D weight, at either width, is fine.
	wantNone(t, "let y = quantize(zeros(4, 8))")
	wantNone(t, "let y = quantize(zeros(4, 8), 4)")
}

func TestGatherConstIndexBoundsCaught(t *testing.T) {
	// A constant index past the first dimension, or a negative one, is the
	// out-of-range error the runtime raises; both the bracket and tensor([...])
	// forms are read, and a whole list is checked position by position.
	wantOne(t, "let y = gather(zeros(3, 2), [7])", "index 7 out of range [0, 3)")
	wantOne(t, "let y = gather(zeros(3, 2), [-1])", "index -1 out of range [0, 3)")
	wantOne(t, "let y = gather(zeros(3, 2), [1, 5])", "index 5 out of range [0, 3)")
	wantOne(t, "let y = gather(zeros(3, 2), tensor([7]))", "index 7 out of range [0, 3)")
	// Indices in range stay clean; a dynamic index is left alone.
	wantNone(t, "let y = gather(zeros(3, 2), [2, 0])")
	wantNone(t, "fn f(i) = gather(zeros(3, 2), i)")
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

func TestTransposeRepeatedAxisCaught(t *testing.T) {
	// A permutation that names an axis twice leaves another unnamed; the runtime
	// rejects it, and the checker now matches its wording.
	wantOne(t, "let y = transpose(zeros(2, 3), 0, 0)", "invalid axis permutation [0 0]")
	wantOne(t, "let y = transpose(zeros(2, 3, 4), 1, 1, 0)", "invalid axis permutation [1 1 0]")
	// A genuine permutation, and the reversing default, stay clean.
	wantNone(t, "let y = transpose(zeros(2, 3), 1, 0)")
	wantNone(t, "let y = transpose(zeros(2, 3, 4), 2, 0, 1)")
}

func TestTopKBoundsCaught(t *testing.T) {
	// k larger than the axis, or a non-positive k, both reached the runtime; so
	// did an out-of-range axis. argtopk shares the same reasoning.
	wantOne(t, "let y = topk(zeros(3), 5)", "k is 5 but axis 0 has length 3")
	wantOne(t, "let y = topk(zeros(3), 0)", "k must be positive")
	wantOne(t, "let y = topk(zeros(2, 4), 5, 0)", "k is 5 but axis 0 has length 2")
	wantOne(t, "let y = topk(zeros(3), 2, 3)", "axis out of range")
	wantOne(t, "let y = argtopk(zeros(3), 9)", "k is 9 but axis 0 has length 3")
	// A k that fits stays clean, on the default and a named axis, and negative axis.
	wantNone(t, "let y = topk(zeros(3), 2)")
	wantNone(t, "let y = topk(zeros(2, 4), 3, 1)")
	wantNone(t, "let y = topk(zeros(2, 4), 2, -1)")
	wantNone(t, "let y = argtopk(zeros(5), 2)")
}

func TestSplitIsAKnownBuiltin(t *testing.T) {
	// split is implemented in both interpreters but was missing from the name
	// tables, so the checker rejected it as an unknown name and it was
	// unreachable. A valid split is now clean.
	wantNone(t, "let x = zeros(6)\nlet p = split(x, 2, 0)")
	wantNone(t, "let x = zeros(6)\nlet p = split(x, [2, 4], 0)")
	wantNone(t, "let x = zeros(2, 6)\nlet p = split(x, 3, 1)")
}

func TestSplitBoundsCaught(t *testing.T) {
	// The mismatches split raises at runtime are static when the axis length and
	// the count or explicit sizes are constant.
	wantOne(t, "let x = zeros(7)\nlet p = split(x, 3, 0)", "does not divide evenly")
	wantOne(t, "let x = zeros(6)\nlet p = split(x, 0, 0)", "count must be positive")
	wantOne(t, "let x = zeros(6)\nlet p = split(x, [2, 2], 0)", "sizes sum to 4 but axis 0 has length 6")
	wantOne(t, "let x = zeros(6)\nlet p = split(x, list(2, 2), 0)", "sizes sum to 4")
	wantOne(t, "let x = zeros(6)\nlet p = split(x, [2, -1], 0)", "negative size -1")
	wantOne(t, "let x = zeros(6)\nlet p = split(x, 2, 5)", "axis out of range")
	// Sizes that add up, and a count that divides, stay clean.
	wantNone(t, "let x = zeros(6)\nlet p = split(x, [2, 4], 0)")
	wantNone(t, "let x = zeros(6)\nlet p = split(x, 3, 0)")
}

func TestMaxpool2dArityCaught(t *testing.T) {
	// The wrong number of arguments reached the runtime; maxpool2d takes exactly
	// the tensor and the window.
	wantOne(t, "let x = zeros(3, 8, 8)\nlet y = maxpool2d(x)", "expects 2 argument(s), got 1")
	wantOne(t, "let x = zeros(3, 8, 8)\nlet y = maxpool2d(x, 2, 3)", "expects 2 argument(s), got 3")
	// The right count stays clean, and a user function of the same name is not the
	// builtin, so its own arity governs.
	wantNone(t, "let x = zeros(3, 8, 8)\nlet y = maxpool2d(x, 2)")
	wantNone(t, "fn maxpool2d(a) = a\nlet y = maxpool2d(5.0)")
}

func TestFixedArityBuiltinsCaught(t *testing.T) {
	// A representative sweep across the arity table: the wrong count is settled
	// statically, mirroring the runtime's "expects N argument(s)".
	wantOne(t, "let y = item(zeros(1), 2)", "item expects 1 argument(s), got 2")
	wantOne(t, "let y = shape()", "shape expects 1 argument(s), got 0")
	wantOne(t, "let y = linear(zeros(2, 3))", "linear expects 2 argument(s), got 1")
	wantOne(t, "let y = clip(zeros(3), 0.0)", "clip expects 3 argument(s), got 2")
	wantOne(t, "let y = where(zeros(3), zeros(3))", "where expects 3 argument(s), got 2")
	// Builtins registered through the interpreter's helper functions (unaryOp,
	// elemOp, bitOp, binTensor) are fixed-arity too and were once absent.
	wantOne(t, "let y = conv2d(zeros(1, 4, 4), zeros(2, 1, 2, 2), 0)", "conv2d expects 2 argument(s), got 3")
	wantOne(t, "let y = relu(zeros(3), 1)", "relu expects 1 argument(s), got 2")
	wantOne(t, "let y = maximum(zeros(3))", "maximum expects 2 argument(s), got 1")
	// Correct counts stay clean; variadic builtins are not constrained.
	wantNone(t, "let y = clip(zeros(3), 0.0, 1.0)")
	wantNone(t, "let y = transpose(zeros(2, 3))")
	wantNone(t, "let y = sum(zeros(2, 3), 0)")
	wantNone(t, "let y = reshape(zeros(6), 2, 3)")
	// A user function shadowing a builtin keeps its own arity, not the table's.
	wantNone(t, "fn item(a, b) = a\nlet y = item(1.0, 2.0)")
}

func TestMaxpool2dWindowCaught(t *testing.T) {
	// A window below one, or one larger than the input's height or width (so the
	// pooled dimension rounds to zero), are the runtime's two window rejections.
	wantOne(t, "let y = maxpool2d(zeros(3, 8, 8), 0)", "window must be >= 1, got 0")
	wantOne(t, "let y = maxpool2d(zeros(3, 4, 4), 9)", "window 9 is larger than input 4x4")
	wantOne(t, "let y = maxpool2d(zeros(3, 8, 4), 5)", "window 5 is larger than input 8x4")
	// A window that fits stays clean, and the pooled shape is now exact.
	wantNone(t, "let y = maxpool2d(zeros(3, 8, 8), 2)")
}

func TestConcatEmptyListCaught(t *testing.T) {
	// An empty list has nothing to join; the runtime needs at least one tensor.
	wantOne(t, "let y = concat([], 0)", "need at least one tensor")
	// concat with a single argument is an arity error before it is an empty one.
	wantOne(t, "let y = concat([])", "expects 2 argument(s), got 1")
	// A non-empty concat stays clean; a dynamic list is left alone (not literal).
	wantNone(t, "let y = concat([zeros(2), ones(3)], 0)")
}

func TestDiffMinLengthCaught(t *testing.T) {
	// Successive differences need two elements along the axis; one leaves nothing
	// to subtract. The axis defaults to the last, so the bare form is checked too.
	wantOne(t, "let y = diff(zeros(3, 1), 1)", "at least 2 elements along axis 1, got 1")
	wantOne(t, "let y = diff(zeros(3, 1))", "at least 2 elements along axis 1, got 1")
	// Two or more along the axis is fine; a different axis with room is fine.
	wantNone(t, "let y = diff(zeros(3, 4), 1)")
	wantNone(t, "let y = diff(zeros(3, 1), 0)")
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
