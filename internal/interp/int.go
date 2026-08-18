package interp

import (
	"fmt"
	"math"
	"strconv"

	"github.com/twill-lang/twill/internal/ast"
	"github.com/twill-lang/twill/internal/tensor"
	"github.com/twill-lang/twill/internal/value"
)

// This file is the whole of the I64 runtime: where an Int comes from, how two
// of them combine, and how one meets a Num. The rules are the ones written in
// docs/language-guide.md under "Bitwise operators on I64" and "Integer division
// and modulo on I64"; nothing here decides semantics, it implements them.

// isF64Anno reports whether an annotation is a bare `F64`, the systems-mode
// float. Like isI64Anno it reads whichever slot the parser used.
func isF64Anno(typeName string, u *ast.UnitAnno) bool {
	if typeName == "F64" {
		return true
	}
	return u != nil && len(u.Factors) == 1 && u.Factors[0].Name == "F64" && u.Factors[0].Exp == 1
}

// coerceScalarAnno applies an `I64` or `F64` annotation to a value: an I64
// annotation makes an Int of a number, an F64 annotation makes a Num of one.
// Anything that is not a scalar number is left alone, because the annotation is
// then a type the checker owns and the runtime has nothing to convert.
//
// A fractional number bound at I64 truncates toward zero, which is what
// `let mid: I64 = (lo + hi) / 2` has always meant here. A number an I64 cannot
// hold -- NaN, an infinity, or a magnitude at or beyond 2^63 -- is an error
// rather than a saturated or wrapped value, because every one of those is a
// mistake upstream and a silently clamped index is the worst way to learn of it.
func (ip *Interp) coerceScalarAnno(typeName string, u *ast.UnitAnno, v value.Value, line int) value.Value {
	if isI64Anno(typeName, u) {
		out, err := toInt(v)
		if err != nil {
			ip.panicf(line, "%s", err.Error())
		}
		return out
	}
	if isF64Anno(typeName, u) {
		if i, ok := v.(value.Int); ok {
			return value.Num(float64(i))
		}
	}
	return v
}

// toInt converts a numeric value to an Int, truncating a fractional one toward
// zero. A non-numeric value passes through unchanged (see coerceScalarAnno).
func toInt(v value.Value) (value.Value, error) {
	switch t := v.(type) {
	case value.Int:
		return t, nil
	case value.Num:
		return intOfFloat(float64(t))
	case *tensor.Tensor:
		if t.IsScalar() {
			return intOfFloat(t.Data[0])
		}
	}
	return v, nil
}

func intOfFloat(f float64) (value.Value, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) || f < -9223372036854775808.0 || f >= 9223372036854775808.0 {
		return nil, fmt.Errorf("cannot convert %s to I64: out of range", value.FormatNumber(f))
	}
	return value.Int(int64(math.Trunc(f))), nil
}

// intLiteral reads an integer literal that an f64 would round. Small integer
// literals stay Num, so a numeric-mode program is untouched; a literal above
// 2^53 becomes an Int, since the only alternative is a value that is not the one
// written. `negate` is set when the literal sits under a unary minus, which is
// how MIN_I64 is spelled: 9223372036854775808 alone does not fit and is
// rejected, but its negation does.
func intLiteral(lit *ast.NumberLit, negate bool) (value.Value, bool) {
	text := lit.Text
	if text == "" {
		return nil, false
	}
	for _, c := range text {
		if c < '0' || c > '9' {
			return nil, false
		}
	}
	if negate {
		text = "-" + text
	}
	n, err := strconv.ParseInt(text, 10, 64)
	if err != nil {
		return nil, false
	}
	const maxExact = 1 << 53
	if n >= -maxExact && n <= maxExact {
		return nil, false
	}
	return value.Int(n), true
}

// intOperands reports whether a binary operation is an I64 operation: both
// sides Int, or one side Int and the other an integral Num standing for one.
// Two Nums are never an I64 operation, whatever they hold, which is what keeps
// numeric mode's `/` a float division.
func intOperands(l, r value.Value) (int64, int64, bool) {
	li, lIsInt := l.(value.Int)
	ri, rIsInt := r.(value.Int)
	if lIsInt && rIsInt {
		return int64(li), int64(ri), true
	}
	if lIsInt {
		if n, ok := value.AsInt(r); ok {
			return int64(li), n, true
		}
		return 0, 0, false
	}
	if rIsInt {
		if n, ok := value.AsInt(l); ok {
			return n, int64(ri), true
		}
	}
	return 0, 0, false
}

// intArith is `+ - * / % //` on two I64s. Everything wraps; the two divisions
// truncate toward zero and refuse a zero divisor. `^` is left to the float path
// on purpose: an integer power that wraps is a bit pattern nobody asked for,
// and shl is the spelling for one that is.
func intArith(op string, x, y int64) (value.Value, error) {
	switch op {
	case "+":
		return value.Int(x + y), nil
	case "-":
		return value.Int(x - y), nil
	case "*":
		return value.Int(x * y), nil
	case "/", "//":
		if y == 0 {
			return nil, fmt.Errorf("integer division by zero")
		}
		// Go's int64 division already gives MIN_I64 / -1 = MIN_I64 without a
		// trap, which is the wrap the guide specifies.
		return value.Int(x / y), nil
	case "%":
		if y == 0 {
			return nil, fmt.Errorf("integer modulo by zero")
		}
		return value.Int(x % y), nil
	}
	return nil, nil
}

// intCompare orders two I64s exactly. It is what a comparison uses when either
// side is an Int, so a value above 2^53 is not compared through an f64 that
// cannot tell it from its neighbour.
func intCompare(op string, x, y int64) (bool, bool) {
	switch op {
	case "==":
		return x == y, true
	case "!=":
		return x != y, true
	case "<":
		return x < y, true
	case "<=":
		return x <= y, true
	case ">":
		return x > y, true
	case ">=":
		return x >= y, true
	}
	return false, false
}
