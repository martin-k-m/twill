package tensor

import (
	"math"
	"strconv"
	"testing"
)

// The golden rounded values were cross-checked against the self-hosted reference
// (src/tensor.tw via `x.to(dt)`) on its positive inputs, where it is correct;
// the negatives mirror the positives here, which is where the reference has a
// sign bug that zeros them (noted for a separate fix).
func TestRoundToDTypeGolden(t *testing.T) {
	cases := []struct {
		dt   DType
		in   float64
		want float64
	}{
		{DTBF16, 0.1, 0.10009765625},
		{DTBF16, 0.3, 0.30078125},
		{DTBF16, -0.3, -0.30078125}, // sign preserved
		{DTBF16, 3.14159, 3.140625},
		{DTBF16, 65505, 65536},   // rounds up past the f16-range value
		{DTBF16, 100000, 99840},  // still finite: bf16 has f32's range
		{DTBF16, 1.0009765625, 1}, // below bf16's resolution near 1
		{DTF16, 0.1, 0.0999755859375},
		{DTF16, 3.14159, 3.140625},
		{DTF16, 65505, 65504},          // the largest finite f16
		{DTF16, 100000, math.Inf(1)},   // overflow to +Inf
		{DTF16, -70000, math.Inf(-1)},  // overflow to -Inf, sign kept
		{DTF16, 1.0009765625, 1.0009765625},
		{DTF32, 0.1, float64(float32(0.1))},
		{DTF32, -0.3, float64(float32(-0.3))},
		{DTF32, 100000, 100000},
		{DTF64, 3.14159, 3.14159}, // exact
	}
	for _, c := range cases {
		if got := RoundToDType(c.dt, c.in); got != c.want {
			t.Errorf("RoundToDType(%s, %g) = %g, want %g", DTypeName(c.dt), c.in, got, c.want)
		}
	}
}

// The integers truncate toward zero and clamp at their range rather than
// wrapping, and a NaN has no integer image so it becomes zero.
func TestRoundToDTypeIntegers(t *testing.T) {
	cases := []struct {
		dt   DType
		in   float64
		want float64
	}{
		{DTI8, 3.9, 3},      // toward zero
		{DTI8, -3.9, -3},    // toward zero, not toward -inf
		{DTI8, 200, 127},    // clamp high, not wrap to -56
		{DTI8, -200, -128},  // clamp low
		{DTI8, math.NaN(), 0},
		{DTI32, 3e9, 2147483647},
		{DTI32, -3e9, -2147483648},
		{DTI32, 5.5, 5},
	}
	for _, c := range cases {
		if got := RoundToDType(c.dt, c.in); got != c.want {
			t.Errorf("RoundToDType(%s, %g) = %g, want %g", DTypeName(c.dt), c.in, got, c.want)
		}
	}
}

// bool is x != 0, so a NaN is true; it is the only reading under which bool(x)
// and x != 0 agree.
func TestRoundToDTypeBool(t *testing.T) {
	for _, in := range []float64{0} {
		if got := RoundToDType(DTBool, in); got != 0 {
			t.Errorf("bool(%g) = %g, want 0", in, got)
		}
	}
	for _, in := range []float64{1, -1, 0.5, 1e-40, math.NaN(), math.Inf(1)} {
		if got := RoundToDType(DTBool, in); got != 1 {
			t.Errorf("bool(%g) = %g, want 1", in, got)
		}
	}
}

// Narrowing rounds to nearest, ties to even: 1 + 3*ulp lands on a tie that rounds
// up to an even last bit, while 1 + 1*ulp rounds down to it.
func TestRoundNarrowTiesToEven(t *testing.T) {
	// bf16 has 7 stored mantissa bits, so its ulp at 1.0 is 2^-7.
	ulp := math.Ldexp(1, -7)
	// 1 + 0.5 ulp is a tie between 1 and 1+ulp; 1 is even, so it rounds down.
	if got := RoundToDType(DTBF16, 1+0.5*ulp); got != 1 {
		t.Errorf("bf16(1 + 0.5ulp) = %v, want 1 (tie to even down)", got)
	}
	// 1 + 1.5 ulp is a tie between 1+ulp and 1+2ulp; the latter is even.
	if got := RoundToDType(DTBF16, 1+1.5*ulp); got != 1+2*ulp {
		t.Errorf("bf16(1 + 1.5ulp) = %v, want %v (tie to even up)", got, 1+2*ulp)
	}
}

// A magnitude far below the smallest normal rounds through the subnormals to
// zero, and the sign of zero is preserved throughout.
func TestRoundNarrowSubnormalAndSignedZero(t *testing.T) {
	if got := RoundToDType(DTF16, 1e-40); got != 0 {
		t.Errorf("f16(1e-40) = %v, want 0 (underflow)", got)
	}
	if got := RoundToDType(DTBF16, math.Copysign(0, -1)); !math.Signbit(got) || got != 0 {
		t.Errorf("bf16(-0) = %v, want -0", got)
	}
}

// NaN and the infinities pass through every float dtype unchanged.
func TestRoundNarrowSpecials(t *testing.T) {
	for _, dt := range []DType{DTF16, DTBF16, DTF32} {
		if got := RoundToDType(dt, math.NaN()); !math.IsNaN(got) {
			t.Errorf("%s(NaN) = %v, want NaN", DTypeName(dt), got)
		}
		if got := RoundToDType(dt, math.Inf(1)); !math.IsInf(got, 1) {
			t.Errorf("%s(+Inf) = %v, want +Inf", DTypeName(dt), got)
		}
	}
}

func TestPromote(t *testing.T) {
	cases := []struct{ a, b, want DType }{
		{DTF32, DTF32, DTF32},   // same
		{DTBool, DTI8, DTI8},    // two ints -> wider; bool is narrowest
		{DTI8, DTI32, DTI32},    // two ints -> wider
		{DTBF16, DTI8, DTBF16},  // int with float -> the float, unchanged
		{DTI32, DTF16, DTF16},   // int with float -> the float
		{DTF16, DTBF16, DTF32},  // neither contains the other -> f32
		{DTBF16, DTF16, DTF32},  // symmetric
		{DTF64, DTF32, DTF64},   // wider float
		{DTF32, DTBF16, DTF32},  // wider float
	}
	for _, c := range cases {
		if got := Promote(c.a, c.b); got != c.want {
			t.Errorf("Promote(%s, %s) = %s, want %s", DTypeName(c.a), DTypeName(c.b), DTypeName(got), DTypeName(c.want))
		}
	}
}

func TestAccDType(t *testing.T) {
	cases := []struct{ dt, want DType }{
		{DTF64, DTF64},
		{DTF32, DTF32},
		{DTBF16, DTF32}, // anything narrower than f32 accumulates in f32
		{DTF16, DTF32},
		{DTI8, DTI32}, // integers accumulate in i32
		{DTBool, DTI32},
		{DTI32, DTI32},
	}
	for _, c := range cases {
		if got := AccDType(c.dt); got != c.want {
			t.Errorf("AccDType(%s) = %s, want %s", DTypeName(c.dt), DTypeName(got), DTypeName(c.want))
		}
	}
}

// A tensor's dtype defaults to f64 (the zero value of the internal field), and
// WithDType stamps another without disturbing the data.
func TestTensorDTypeField(t *testing.T) {
	x := New([]float64{1, 2, 3}, []int{3})
	if x.DType() != DTF64 {
		t.Errorf("a fresh tensor is %s, want f64", DTypeName(x.DType()))
	}
	// A raw intermediate (no constructor) is f64 too, via the zero value.
	if (&Tensor{Data: []float64{0}, Shape: []int{1}}).DType() != DTF64 {
		t.Error("a bare &Tensor{} should read as f64")
	}
	x.WithDType(DTBF16)
	if x.DType() != DTBF16 {
		t.Errorf("after WithDType(bf16) got %s", DTypeName(x.DType()))
	}
	// bool is code 0; the plus-one encoding must not confuse it with the default.
	b := New([]float64{1, 0}, []int{2}).WithDType(DTBool)
	if b.DType() != DTBool {
		t.Errorf("bool tensor read back as %s", DTypeName(b.DType()))
	}
}

// Cast rounds every element to the target dtype, tags the result, and keeps the
// shape. The negatives round correctly here, which is where the self-hosted
// reference has its NEEDS-2 sign bug, so this is Go-only.
func TestCast(t *testing.T) {
	x := New([]float64{0.1, 0.3, -0.3, 65505}, []int{4})
	y := Cast(x, DTBF16)
	if y.DType() != DTBF16 {
		t.Errorf("cast result is %s, want bf16", DTypeName(y.DType()))
	}
	want := []float64{0.10009765625, 0.30078125, -0.30078125, 65536}
	for i, w := range want {
		if y.Data[i] != w {
			t.Errorf("cast[%d] = %g, want %g", i, y.Data[i], w)
		}
	}
	if len(y.Shape) != 1 || y.Shape[0] != 4 {
		t.Errorf("cast changed the shape to %v", y.Shape)
	}
	// The source is untouched: a cast is not a mutation.
	if x.DType() != DTF64 || x.Data[2] != -0.3 {
		t.Error("cast mutated its input")
	}
}

func TestShortestForDType(t *testing.T) {
	// The stored bf16 value of 0.1 spells back to "0.1"; whole and integer dtypes
	// print at natural length; the specials keep Go's spellings.
	cases := []struct {
		dt   DType
		in   float64
		want string
	}{
		{DTBF16, RoundToDType(DTBF16, 0.1), "0.1"},
		{DTF16, RoundToDType(DTF16, 3.14159), "3.14"},
		{DTBF16, 1, "1"},
		{DTI8, 3, "3"},
		{DTBF16, math.Inf(1), "+Inf"},
		{DTF16, math.NaN(), "NaN"},
	}
	for _, c := range cases {
		if got := ShortestForDType(c.dt, c.in); got != c.want {
			t.Errorf("ShortestForDType(%s, %g) = %q, want %q", DTypeName(c.dt), c.in, got, c.want)
		}
	}
	// The shortest spelling really round-trips: parsed and re-rounded, it is the
	// stored value again, for a sweep of bf16 and f16 values.
	for _, x := range []float64{0.3, 2.5, 100, 0.333, 65504, 1.0009765625} {
		for _, dt := range []DType{DTBF16, DTF16} {
			stored := RoundToDType(dt, x)
			s := ShortestForDType(dt, stored)
			back, err := strconv.ParseFloat(s, 64)
			if err != nil || RoundToDType(dt, back) != stored {
				t.Errorf("%s %g -> %q does not round-trip", DTypeName(dt), x, s)
			}
		}
	}
}

func TestDTypeNames(t *testing.T) {
	for _, dt := range []DType{DTBool, DTI8, DTI32, DTF16, DTBF16, DTF32, DTF64} {
		name := DTypeName(dt)
		if got, ok := DTypeOfName(name); !ok || got != dt {
			t.Errorf("round trip %s -> %v (ok=%v)", name, got, ok)
		}
	}
	if _, ok := DTypeOfName("f8"); ok {
		t.Error("f8 is not a dtype but was accepted")
	}
}
