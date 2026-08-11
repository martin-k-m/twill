package interp_test

import "testing"

// str_to_f64 and f64_to_str are the native float text conversions the
// self-hosted lexer and formatter delegate to, so the bootstrap reads and
// prints numeric literals exactly rather than through the pure-twill decimal
// machinery, which needs an exact 64-bit integer the float64 runtime lacks.
func TestStrToF64RoundTrips(t *testing.T) {
	cases := map[string]string{
		"7":     "7",
		"42":    "42",
		"3.5":   "3.5",
		"-0.5":  "-0.5",
		"0.05":  "0.05",
		"1e-16": "1e-16",
	}
	for in, want := range cases {
		// Print through f64_to_str (%g), not str (%.6f), so a tiny magnitude like
		// 1e-16 is shown by the value it parsed to rather than rounded to "0".
		src := "let r = str_to_f64(\"" + in + "\")\nmatch r { Some(v) => print(f64_to_str(v)), None => print(\"none\") }"
		if got := runOut(t, "mode systems\n"+src); got != want {
			t.Errorf("str_to_f64(%q) printed %q, want %q", in, got, want)
		}
	}
}

func TestF64ToStrShortest(t *testing.T) {
	cases := map[string]string{
		"0.5":   "0.5",
		"1.5":   "1.5",
		"-0.25": "-0.25",
	}
	for in, want := range cases {
		if got := runOut(t, "mode systems\nprint(f64_to_str("+in+"))"); got != want {
			t.Errorf("f64_to_str(%s) = %q, want %q", in, got, want)
		}
	}
}
