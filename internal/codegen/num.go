package codegen

import (
	"math"
	"strconv"
)

// hexFloat prints a float64 as a C99 hexadecimal floating literal, which round
// trips exactly. Printing constants in decimal would put the emitted code's
// numerics at the mercy of the C compiler's decimal-to-binary conversion, and a
// backend that claims bit-exact agreement cannot afford a constant that differs
// in the last place from the one the interpreter used.
func hexFloat(x float64) string {
	switch {
	case math.IsNaN(x):
		return "NAN"
	case math.IsInf(x, 1):
		return "INFINITY"
	case math.IsInf(x, -1):
		return "(-INFINITY)"
	}
	s := strconv.FormatFloat(x, 'x', -1, 64)
	if s[0] == '-' {
		return "(" + s + ")"
	}
	return s
}
