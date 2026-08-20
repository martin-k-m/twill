package checker_test

import "testing"

// The one warning this checker emits (NEEDS-113): a narrow float silently
// widened by a wider one. `bf16_weights + f64_bias` is f64, which is a
// perfectly correct answer that undoes the reason the weights were narrow, and
// the checker is the only place it can be seen before the program runs.
//
// The message is byte-identical to src/check.tw's promote_and_warn, because a
// differential harness compares the two checkers character for character.

func TestWideningIsReported(t *testing.T) {
	wantOne(t, "mode systems\nfn f() {\n  let w = zeros(2, 3, bf16)\n  let y = w * scalar(2.0)\n}",
		"dtype widening: bf16 and f64 promote to f64, which undoes the reason the bf16 operand is narrow")
}

// A cast is the other way a dtype is stated, and `@` promotes like the rest.
func TestWideningThroughACastAndMatmul(t *testing.T) {
	wantOne(t, "mode systems\nfn f() {\n  let a = zeros(2, 3).to(f16)\n  let b = zeros(3, 2).to(f32)\n  let y = a @ b\n}",
		"dtype widening: f16 and f32 promote to f32, which undoes the reason the f16 operand is narrow")
}

// Same dtype on both sides is not a widening.
func TestSameDTypeIsSilent(t *testing.T) {
	wantNone(t, "mode systems\nfn f() {\n  let a = zeros(2, 3, bf16)\n  let b = ones(2, 3).to(bf16)\n  let y = a * b\n}")
}

// f16 meeting bf16 promotes past both to f32, so NEITHER operand was the wider
// one and there is nothing to tell the author to change.
func TestNeitherOperandWiderIsSilent(t *testing.T) {
	wantNone(t, "mode systems\nfn f() {\n  let a = zeros(2, 3, f16)\n  let b = zeros(2, 3, bf16)\n  let y = a * b\n}")
}

// An integer meeting a float keeps the float unchanged: the lattice doing its
// job, not a loss.
func TestIntMeetingFloatIsSilent(t *testing.T) {
	wantNone(t, "mode systems\nfn f() {\n  let a = zeros(2, 3, i8)\n  let b = zeros(2, 3, bf16)\n  let y = a * b\n}")
}

// THE HARDER PROMISE: zero new diagnostics on a program that never wrote a
// dtype. A bare number literal deliberately has no dtype -- only `scalar(x)`
// and a tensor literal are f64 -- so ordinary arithmetic stays silent.
func TestDTypeFreeProgramIsSilent(t *testing.T) {
	wantNone(t, "mode systems\nfn f() {\n  let a = zeros(2, 3)\n  let b = ones(2, 3)\n  let y = a * b + 1.0\n}")
}

// An integer input to a float-only operation degrades to unknown rather than
// claiming f32, so a chain like exp(argmax(x)) cannot make the warning fire on
// a program that never wrote a dtype.
func TestFloatOnlyOpOfAnIntDegradesToUnknown(t *testing.T) {
	wantNone(t, "mode systems\nfn f() {\n  let idx = argmax(zeros(2, 3))\n  let y = exp(idx) * scalar(2.0)\n}")
}

// A constructor's trailing dtype is contextual: it counts only when nothing in
// scope binds the name. Bound, it is an ordinary value again -- and the shape
// must still come from every argument before it.
func TestMakerKeepsItsShapeAlongsideADType(t *testing.T) {
	wantOne(t, "mode systems\nfn f() {\n  let a = zeros(2, 3, bf16)\n  let y = a * zeros(9, 9, bf16)\n}",
		"shape mismatch: [2, 3] vs [9, 9] cannot broadcast")
}
