package interp_test

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/twill-lang/twill/internal/interp"
)

// I64 is a real 64-bit integer as of 1.6: docs/needs.md NEEDS-2, and the
// semantics written down in docs/language-guide.md under "Bitwise operators on
// I64" and "Integer division and modulo on I64". Every table here is that page.

func runI64(t *testing.T, body string) []string {
	t.Helper()
	return runFile(t, t.TempDir(), "mode systems\n"+body)
}

// An I64 holds every value in its range exactly. The f64 it used to be carried
// held 53 bits, so 2^53 + 1 was 2^53 and every hash mixer in the ecosystem was
// silently wrong.
func TestI64IsExactAbove2To53(t *testing.T) {
	out := runI64(t, `
let x: I64 = 9007199254740993
print(x)
print(x == x + 1)
print(x - 1 == 9007199254740992)
let mx: I64 = 9223372036854775807
let mn: I64 = -9223372036854775808
print(mx)
print(mn)
print(mx > mx - 1)
print(mn < mn + 1)
print(1234567890123456789)
print(9007199254740993 == 9007199254740992)
`)
	expectLines(t, out, "9007199254740993", "false", "true",
		"9223372036854775807", "-9223372036854775808", "true", "true",
		"1234567890123456789", "false")
}

// Arithmetic wraps rather than trapping, including the one division that
// overflows.
func TestI64ArithmeticWraps(t *testing.T) {
	out := runI64(t, `
let mx: I64 = 9223372036854775807
let mn: I64 = -9223372036854775808
print(mx + 1)
print(mn - 1)
print(mx * 2)
print(mn / -1)
print(mn % -1)
let a: I64 = 6364136223846793005
let s: I64 = 42
print(s * a + 1442695040888963407)
`)
	expectLines(t, out, "-9223372036854775808", "9223372036854775807", "-2",
		"-9223372036854775808", "0", "-7964744663189004623")
}

// `/` truncates toward zero and `%` takes the sign of the dividend, so that
// (a / b) * b + a % b == a. Numeric mode's `%` is floored and disagrees; the
// two rules are deliberate (docs/language-guide.md).
func TestI64DivisionAndModuloTruncate(t *testing.T) {
	out := runI64(t, `
let p: I64 = 7
let n: I64 = -7
let two: I64 = 2
let mtwo: I64 = -2
print(p / two, p % two)
print(n / two, n % two)
print(p / mtwo, p % mtwo)
print(n / mtwo, n % mtwo)
print(n // two, n % 3)
print((n / two) * two + n % two == n)
`)
	expectLines(t, out, "3 1", "-3 -1", "-3 1", "3 -1", "-3 -1", "true")

	// Numeric mode is untouched: `%` is still floored there and `/` is exact.
	out = runFile(t, t.TempDir(), "print(-7 % 3)\nprint(7 / 2)\nprint(-7 // 2)\n")
	expectLines(t, out, "2", "3.5", "-3")
}

func TestI64DivisionByZeroIsAnError(t *testing.T) {
	for _, op := range []string{"/", "%", "//"} {
		src := fmt.Sprintf("mode systems\nlet a: I64 = 1\nlet z: I64 = 0\nprint(a %s z)\n", op)
		_, err := runSrcErr(t, src)
		if err == nil || !strings.Contains(err.Error(), "by zero") {
			t.Errorf("%s: err = %v, want a division-by-zero error", op, err)
		}
	}
}

// The bitwise words are exact on all 64 bits: the sign bit is a bit like any
// other, shifts mask their count to 0..63, and shr is arithmetic.
func TestI64BitwiseIsExact(t *testing.T) {
	out := runI64(t, `
let mx: I64 = 9223372036854775807
let mn: I64 = -9223372036854775808
print(shl(1, 63))
print(shl(1, 64))
print(shr(mx, 63))
print(shr(mn, 63))
print(shr(-8, 1), shr(-1, 70))
print(bnot(0), bnot(mx))
print(band(mx, 255), bor(mn, 1))
print(xor(mx, mn))
print(mx band 255, 1 shl 62)
`)
	expectLines(t, out, "-9223372036854775808", "1", "0", "-1", "-4 -1",
		"-1 -9223372036854775808", "255 -9223372036854775807", "-1", "255 4611686018427387904")
}

// The conversions across the seam: i64 truncates toward zero, f64 widens.
// An I64 annotation on a binding, a parameter, a return or a struct field
// converts, and an F64 annotation converts back.
func TestI64ConversionsAndAnnotations(t *testing.T) {
	out := runI64(t, `
struct Counter { n: I64, scale: F64 }
fn ceil_div(n: I64, k: I64) -> I64 { (n + k - 1) / k }
fn half(n: I64) -> F64 { f64(n) / 2 }
fn takes_int(x: I64) -> I64 { x / 2 }
let c: Counter = Counter { n: 7, scale: 3 }
print(i64(3.9), i64(-3.9), i64(9007199254740993))
print(f64(9223372036854775807))
print(ceil_div(7, 2))
print(half(7))
print(takes_int(7.0))
print(c.n / 2)
print(c.scale / 2)
let big: I64 = 9007199254740993
let widened: F64 = big
print(widened)
print(str(big) + "!")
`)
	expectLines(t, out, "3 -3 9007199254740993", "9223372036854775808", "4", "3.5", "3",
		"3", "1.5", "9007199254740992", "9007199254740993!")
}

// An I64 meeting a fractional number widens to F64, which is what the same
// expression computed before Int existed; an I64 meeting a whole number is an
// I64 operation.
func TestI64MixedWithNumber(t *testing.T) {
	out := runI64(t, `
let n: I64 = 7
print(n * 1.5)
print(n + 1)
print(n / 2)
print(n == 7.0, n == 7.5, n < 7.5)
let d: Dict[I64, Str] = {}
dict_set(d, 9007199254740993, "big")
print(dict_get(d, 9007199254740993), dict_get(d, 9007199254740992))
`)
	expectLines(t, out, "10.5", "8", "3", "true false true", "Some(big) None")
}

// A number an I64 cannot hold is an error at the annotation, not a clamped
// value.
func TestI64OutOfRangeIsAnError(t *testing.T) {
	_, err := runSrcErr(t, "mode systems\nlet x: I64 = 1e300\n")
	if err == nil || !strings.Contains(err.Error(), "cannot convert") {
		t.Fatalf("err = %v, want a conversion error", err)
	}
}

// The property test: random I64 pairs through every operator, against Go's
// int64, which has the same wrapping and truncation rules.
func TestI64PropertyAgainstGo(t *testing.T) {
	rng := rand.New(rand.NewSource(1606))
	pick := func() int64 {
		switch rng.Intn(6) {
		case 0:
			return 9223372036854775807
		case 1:
			return -9223372036854775808
		case 2:
			return int64(rng.Intn(21) - 10)
		case 3:
			return int64(rng.Uint64() >> uint(rng.Intn(64)))
		default:
			return int64(rng.Uint64())
		}
	}
	var sb strings.Builder
	sb.WriteString("mode systems\n")
	var want []string
	for i := 0; i < 300; i++ {
		a, b := pick(), pick()
		k := int64(rng.Intn(70))
		fmt.Fprintf(&sb, "let a%d: I64 = %d\nlet b%d: I64 = %d\n", i, a, i, b)
		fmt.Fprintf(&sb, "print(a%d + b%d, a%d - b%d, a%d * b%d, band(a%d, b%d), bor(a%d, b%d), xor(a%d, b%d), shl(a%d, %d), shr(a%d, %d), a%d < b%d)\n",
			i, i, i, i, i, i, i, i, i, i, i, i, i, k, i, k, i, i)
		want = append(want, fmt.Sprintf("%d %d %d %d %d %d %d %d %v",
			a+b, a-b, a*b, a&b, a|b, a^b, a<<uint(k&63), a>>uint(k&63), a < b))
		if b != 0 {
			fmt.Fprintf(&sb, "print(a%d / b%d, a%d %% b%d)\n", i, i, i, i)
			want = append(want, fmt.Sprintf("%d %d", a/b, a%b))
		}
	}
	out := runFile(t, t.TempDir(), sb.String())
	if len(out) != len(want) {
		t.Fatalf("got %d lines, want %d", len(out), len(want))
	}
	for i := range want {
		if out[i] != want[i] {
			t.Errorf("line %d: got %q want %q", i, out[i], want[i])
		}
	}
}

func runSrcErr(t *testing.T, src string) ([]string, error) {
	t.Helper()
	var out []string
	ip := interp.New(func(s string) { out = append(out, s) })
	_, err := ip.Run(src)
	return out, err
}
