package tensor

import (
	"math"
	"strconv"
)

// DType is a tensor element type. The numeric codes match src/tensor.tw's DT_*
// constants exactly, so the two implementations name the same dtype by the same
// integer; docs/dtypes.md is the design. This file is the numerics only -- which
// values a dtype holds, what a store rounds to, what a mixed operation promotes
// to, and what an operation class accumulates in. It deliberately touches
// nothing else: no Tensor field, no kernel, no formatter. Those are later steps
// and this is the part docs/dtypes.md calls "the part that is wrong in most
// implementations", so it is landed and tested on its own first.
type DType int

const (
	DTBool DType = 0
	DTI8   DType = 1
	DTI32  DType = 2
	DTF16  DType = 3
	DTBF16 DType = 4
	DTF32  DType = 5
	DTF64  DType = 6
)

// dtypeName is the surface spelling of each dtype, and dtypeCode is its inverse.
// The seven names are the only ones the language accepts (docs/dtypes.md).
var dtypeName = map[DType]string{
	DTBool: "bool", DTI8: "i8", DTI32: "i32", DTF16: "f16",
	DTBF16: "bf16", DTF32: "f32", DTF64: "f64",
}

// DTypeName returns the surface spelling of a dtype, or "" for an unknown code.
func DTypeName(dt DType) string { return dtypeName[dt] }

// DTypeOfName maps a surface spelling to its dtype, reporting ok=false for a
// string that is not one of the seven.
func DTypeOfName(s string) (DType, bool) {
	switch s {
	case "bool":
		return DTBool, true
	case "i8":
		return DTI8, true
	case "i32":
		return DTI32, true
	case "f16":
		return DTF16, true
	case "bf16":
		return DTBF16, true
	case "f32":
		return DTF32, true
	case "f64":
		return DTF64, true
	}
	return 0, false
}

// DTypeBits is a dtype's storage width. bool and i8 are eight bits, not one and
// four, because a sub-byte element needs bit addressing that is not worth it yet
// (docs/dtypes.md).
func DTypeBits(dt DType) int {
	switch dt {
	case DTBool, DTI8:
		return 8
	case DTF16, DTBF16:
		return 16
	case DTI32, DTF32:
		return 32
	default:
		return 64
	}
}

// isFloatDType and isIntDType partition the dtypes the way the promotion rules
// need. The codes are ordered so that a single comparison decides each: the
// integers are the three lowest, the floats the four highest.
func isFloatDType(dt DType) bool { return dt >= DTF16 }
func isIntDType(dt DType) bool    { return dt <= DTI32 }

// truncTowardZero is float-to-integer truncation: toward zero, so that
// i32(-0.5) is 0 and not -1. It matches Go's, C's, and every array library's
// float-to-int conversion.
func truncTowardZero(x float64) float64 {
	if x < 0 {
		return -math.Floor(-x)
	}
	return math.Floor(x)
}

// RoundToDType returns the value x takes when stored in dt and read back -- the
// whole observable content of a dtype. Every write into a narrow buffer goes
// through it. It mirrors src/tensor.tw's dt_round bit for bit:
//
//   - f64 is exact.
//   - f32 rounds via the hardware's round-to-nearest-even (float64(float32(x))).
//   - bf16 and f16 round to nearest, ties to even, with overflow to an infinity
//     and correct subnormals; roundNarrowFloat does the bit work.
//   - bool is x != 0, so NaN is true, matching Go's != and making bool(x) agree
//     with x != 0.
//   - the integers truncate toward zero and clamp (not wrap) at their range; a
//     NaN has no integer image and becomes 0 rather than propagating.
func RoundToDType(dt DType, x float64) float64 {
	switch dt {
	case DTF64:
		return x
	case DTF32:
		return float64(float32(x))
	case DTBF16:
		return roundNarrowFloat(x, 8, 7)
	case DTF16:
		return roundNarrowFloat(x, 5, 10)
	case DTBool:
		if x == 0 {
			return 0
		}
		return 1
	default: // DTI8, DTI32
		if math.IsNaN(x) {
			return 0
		}
		lo, hi := -128.0, 127.0
		if dt == DTI32 {
			lo, hi = -2147483648.0, 2147483647.0
		}
		v := truncTowardZero(x)
		if v < lo {
			return lo
		}
		if v > hi {
			return hi
		}
		return v
	}
}

// roundNarrowFloat rounds a float64 to the nearest value of an IEEE-754 binary
// float with expBits exponent bits and mantBits stored mantissa bits, then
// widens it back to float64 -- exactly, since widening never rounds. It is
// round-to-nearest, ties-to-even, with a value above the format's largest finite
// becoming an infinity of the right sign, correct gradual underflow into the
// subnormals, and NaN and the infinities preserved. bf16 is (8, 7) and f16 is
// (5, 10); f32 is left to the hardware in RoundToDType.
func roundNarrowFloat(x float64, expBits, mantBits uint) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return x
	}
	if x == 0 {
		return x // preserves signed zero
	}

	bias := (1 << (expBits - 1)) - 1        // the format's exponent bias
	maxExp := bias                          // unbiased exponent of the largest finite (2^bias * ...)
	minNormalExp := 1 - bias                // unbiased exponent of the smallest normal

	neg := math.Signbit(x)
	ax := math.Abs(x)

	// math.Frexp gives ax = f * 2^e with f in [0.5, 1); the leading one of the
	// 1.mantissa form therefore sits at unbiased exponent e-1.
	_, e := math.Frexp(ax)
	exp := e - 1

	var rounded float64
	if exp < minNormalExp {
		// Subnormal (or below): round ax to a multiple of the smallest subnormal,
		// 2^(minNormalExp - mantBits), ties to even.
		ulp := math.Ldexp(1, minNormalExp-int(mantBits))
		q := ax / ulp
		rounded = math.RoundToEven(q) * ulp
	} else {
		// Normal: keep mantBits after the implicit leading one. Scale so the kept
		// bits are integral, round ties-to-even, scale back.
		scale := math.Ldexp(1, int(mantBits)-exp)
		m := math.RoundToEven(ax * scale)
		rounded = m / scale
	}

	// Overflow: a magnitude at or above 2^(maxExp+1) has no finite image.
	if rounded >= math.Ldexp(1, maxExp+1) {
		rounded = math.Inf(1)
	}
	if neg {
		return -rounded
	}
	return rounded
}

// Promote returns the dtype a mixed operation produces, widening only, following
// docs/dtypes.md's rules in order: same dtype stays; two integers give the
// wider (bool is the narrowest integer); an integer with a float gives the float
// unchanged, so bf16*2 stays bf16; f16 with bf16 gives f32, since neither
// contains the other; otherwise the wider float. It mirrors src/tensor.tw's
// promote.
func Promote(a, b DType) DType {
	if a == b {
		return a
	}
	if isIntDType(a) && isIntDType(b) {
		if a > b {
			return a
		}
		return b
	}
	if isIntDType(a) {
		return b
	}
	if isIntDType(b) {
		return a
	}
	if (a == DTF16 && b == DTBF16) || (a == DTBF16 && b == DTF16) {
		return DTF32
	}
	if a > b {
		return a
	}
	return b
}

// Promote3 folds Promote over three dtypes.
func Promote3(a, b, c DType) DType { return Promote(Promote(a, b), c) }

// ShortestForDType renders a stored element at its dtype: the shortest decimal
// that reads back to the same value once rounded to dt. A narrow element is held
// as the f64 widening of a value that distinguishes only a few digits, so the
// full f64 spelling would claim a precision it does not have (docs/dtypes.md,
// NEEDS-114); this cuts the decimal where no other value of the dtype could
// round-trip from it. Integer-valued elements (bool, i8, i32, and whole floats)
// print at their natural length. It is a display helper -- f64 is rendered by
// the value package's FormatNumber and never reaches here.
func ShortestForDType(dt DType, x float64) string {
	if math.IsNaN(x) {
		return "NaN"
	}
	if math.IsInf(x, 1) {
		return "+Inf"
	}
	if math.IsInf(x, -1) {
		return "-Inf"
	}
	if x == math.Trunc(x) && math.Abs(x) < 1e15 {
		return strconv.FormatInt(int64(x), 10)
	}
	// The shortest 'g' spelling whose value, rounded back to dt, is x again.
	for p := 1; p <= 17; p++ {
		s := strconv.FormatFloat(x, 'g', p, 64)
		if back, err := strconv.ParseFloat(s, 64); err == nil && RoundToDType(dt, back) == x {
			return s
		}
	}
	return strconv.FormatFloat(x, 'g', -1, 64)
}

// reduceResultDType gives a sum/mean's result dtype: the input's, except that a
// mean of an integer is not an integer and promotes to f32 (docs/dtypes.md).
func reduceResultDType(dt DType, mean bool) DType {
	if mean && isIntDType(dt) {
		return DTF32
	}
	return dt
}

// blockSumAcc sums data at the accumulation dtype acc, rounding each step, in the
// same fixed-block order the f64 parallelSum uses so a narrow reduction is
// deterministic and matches the self-hosted block_sum. Below minParallel it is a
// plain running sum; at or above, fixed sumChunk-element blocks are summed and
// their partials combined in block order. Callers use this only for narrow
// dtypes; f64 stays on parallelSum, whose last-bit order the goldens depend on.
func blockSumAcc(data []float64, acc DType) float64 {
	n := len(data)
	if n < minParallel {
		s := 0.0
		for _, x := range data {
			s = RoundToDType(acc, s+x)
		}
		return s
	}
	total := 0.0
	for start := 0; start < n; start += sumChunk {
		end := start + sumChunk
		if end > n {
			end = n
		}
		s := 0.0
		for i := start; i < end; i++ {
			s = RoundToDType(acc, s+data[i])
		}
		total = RoundToDType(acc, total+s)
	}
	return total
}

// unaryResultDType gives a unary op's result dtype: a float input keeps its
// dtype, and an integer input keeps it only for the ops that preserve
// integrality (neg, relu, square, clip); the transcendentals promote it to f32,
// the rule numpy applies (docs/dtypes.md).
func unaryResultDType(preservesInt bool, dt DType) DType {
	if isFloatDType(dt) || preservesInt {
		return dt
	}
	return DTF32
}

// Cast rounds every element to dt, once from the source value, and tags the
// result -- the value each element takes when stored in dt and read back
// (docs/dtypes.md, "The cast is spelled .to(dt)"). It carries a straight-through
// gradient: the vjp copies the incoming gradient unchanged, so a cast inside a
// differentiated function -- a bf16 weight with an f32 master -- does not detach
// its input. Casting from a narrow dtype to another goes directly and not
// through f64, because the buffer already holds the source value's f64 widening,
// so RoundToDType rounds it once.
func Cast(t *Tensor, dt DType) *Tensor {
	data := make([]float64, len(t.Data))
	for i, x := range t.Data {
		data[i] = RoundToDType(dt, x)
	}
	res := (&Tensor{Data: data, Shape: append([]int(nil), t.Shape...)}).WithDType(dt)
	if recordJets && t.RequiresGrad {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			copy(res.jet.d, t.jet.d)
			copy(res.jet.dd, t.jet.dd)
		}
	}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for i := range gt {
			gt[i] += res.Grad[i]
		}
	})
}

// AccDType is the dtype an operation class accumulates in, given the dtype it
// stores. The one rule that decides whether narrow dtypes work at all: anything
// narrower than f32 accumulates in f32, f32 in f32, f64 in f64, and the integers
// in i32 -- which is what stops an i8 dot product overflowing on its fourth
// term. Stated per operation class, never per tensor (docs/dtypes.md).
func AccDType(dt DType) DType {
	if dt == DTF64 {
		return DTF64
	}
	if isFloatDType(dt) {
		return DTF32
	}
	return DTI32
}
