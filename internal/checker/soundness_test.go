package checker_test

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/martin-k-m/twill/internal/checker"
	"github.com/martin-k-m/twill/internal/interp"
	"github.com/martin-k-m/twill/internal/parser"
)

// Differential testing of the checker against the interpreter.
//
// The checker makes one claim and declines to make the other. It claims that
// every diagnostic it reports is a real mistake, and it explicitly does not
// claim that a clean check means the program runs. This file tests the claim it
// makes, over generated programs rather than hand-written ones, and measures the
// size of the gap it declines to close.
//
// The generator produces small programs built from operations whose shape rules
// are fully determined by literal sizes, so the checker has everything it needs
// to decide and a disagreement is a real disagreement rather than the checker
// declining to guess. Dimensions are drawn from a small set, so roughly half the
// programs are shape errors and half are not, and both directions get exercised.
//
// Two outcomes are possible for each program, and they are not symmetric:
//
//   - The checker reports a diagnostic and the program runs. This is a false
//     positive, it breaks the claim, and it fails the test.
//   - The checker is silent and the program fails at runtime. This is a false
//     negative. It is permitted by design, and it is counted and reported rather
//     than failed, so the number is visible instead of merely admitted to.

// dims is deliberately small. Drawing from {1, 2, 3, 4} makes conforming and
// non-conforming pairs about equally likely, and includes 1, which broadcasts
// against anything and is the case a naive equality rule gets wrong.
var dims = []int{1, 2, 3, 4}

type gen struct{ r *rand.Rand }

func (g *gen) dim() int { return dims[g.r.Intn(len(dims))] }

// matrix emits a constructor with literal dimensions, which is the form the
// checker can fold to a concrete shape.
func (g *gen) matrix(rows, cols int) string {
	return fmt.Sprintf("zeros(%d, %d)", rows, cols)
}

func (g *gen) vector(n int) string { return fmt.Sprintf("ones(%d)", n) }

// program builds one random program and returns its source.
func (g *gen) program() string {
	var b strings.Builder
	switch g.r.Intn(8) {
	case 0: // matmul of two matrices
		m, k, k2, n := g.dim(), g.dim(), g.dim(), g.dim()
		fmt.Fprintf(&b, "let A = %s\nlet B = %s\nlet C = A @ B\nprint(sum(C))\n",
			g.matrix(m, k), g.matrix(k2, n))
	case 1: // matrix times vector
		m, k, n := g.dim(), g.dim(), g.dim()
		fmt.Fprintf(&b, "let A = %s\nlet x = %s\nprint(sum(A @ x))\n",
			g.matrix(m, k), g.vector(n))
	case 2: // elementwise add of two matrices, where broadcasting may or may not apply
		a, bb, c, d := g.dim(), g.dim(), g.dim(), g.dim()
		fmt.Fprintf(&b, "let A = %s\nlet B = %s\nprint(sum(A + B))\n",
			g.matrix(a, bb), g.matrix(c, d))
	case 3: // a row vector broadcast across a matrix
		m, k, n := g.dim(), g.dim(), g.dim()
		fmt.Fprintf(&b, "let A = %s\nlet r = %s\nprint(sum(A * r))\n",
			g.matrix(m, k), g.vector(n))
	case 4: // reshape to a possibly wrong element count
		m, k, p, q := g.dim(), g.dim(), g.dim(), g.dim()
		fmt.Fprintf(&b, "let A = %s\nprint(sum(reshape(A, %d, %d)))\n", g.matrix(m, k), p, q)
	case 5: // concat along an axis whose other extents may not agree
		m, k, p, q := g.dim(), g.dim(), g.dim(), g.dim()
		fmt.Fprintf(&b, "let A = %s\nlet B = %s\nprint(sum(concat([A, B], 0)))\n",
			g.matrix(m, k), g.matrix(p, q))
	case 6: // an annotated function called with a possibly wrong argument
		m, k, n := g.dim(), g.dim(), g.dim()
		fmt.Fprintf(&b, "fn f(A: [%d, %d], x: [%d]) -> [%d] { A @ x }\nprint(sum(f(%s, %s)))\n",
			m, k, k, m, g.matrix(m, k), g.vector(n))
	case 7: // a chain, so a mismatch may be several operations from its cause
		m, k, n, p := g.dim(), g.dim(), g.dim(), g.dim()
		fmt.Fprintf(&b, "let A = %s\nlet B = %s\nlet C = A @ B\nlet r = %s\nprint(sum(relu(C) + r))\n",
			g.matrix(m, k), g.matrix(k, n), g.vector(p))
	}
	return b.String()
}

// runs reports whether the program evaluates without error.
func runs(src string) error {
	ip := interp.New(func(string) {})
	_, err := ip.Run(src)
	return err
}

func TestCheckerReportsNoFalsePositives(t *testing.T) {
	// Fixed seed: a differential test that finds a disagreement should find it
	// again on the next run, and a corpus that changes every run cannot be
	// bisected against.
	g := &gen{r: rand.New(rand.NewSource(20260815))}

	const n = 4000
	var rejected, accepted, falseNegatives int
	var firstFalseNegative string

	for i := 0; i < n; i++ {
		src := g.program()
		prog, perr := parser.Parse(src)
		if perr != nil {
			t.Fatalf("generated program does not parse, which is a bug in the generator:\n%s\n%v", src, perr)
		}
		diags := checker.Check(prog)
		err := runs(src)

		switch {
		case len(diags) > 0:
			rejected++
			if err == nil {
				// The claim broken. This is the outcome the test exists to catch.
				t.Errorf("false positive: the checker reported %q but the program ran clean:\n%s",
					diags[0].Msg, src)
			}
		case err != nil:
			falseNegatives++
			if firstFalseNegative == "" {
				firstFalseNegative = fmt.Sprintf("%s\n  runtime error: %v", src, err)
			}
		default:
			accepted++
		}
	}

	t.Logf("checker soundness over %d generated programs:", n)
	t.Logf("  rejected by the checker, and every one of them really is broken: %d", rejected)
	t.Logf("  accepted and ran clean:                                          %d", accepted)
	t.Logf("  accepted but failed at runtime (false negatives):                %d", falseNegatives)
	if firstFalseNegative != "" {
		t.Logf("  first false negative:\n%s", firstFalseNegative)
	}
}

// TestCheckerCatchesWhatItCanSee is the completeness measurement for the subset
// of programs where the checker has everything it needs. Every program the
// generator produces has literal dimensions throughout, so a shape error in one
// is statically decidable in principle. This records what fraction the checker
// actually decides, which is the honest form of "best-effort": a number rather
// than an adjective.
//
// It asserts a floor rather than an exact figure, so that improving the checker
// does not break the test and regressing it does.
func TestCheckerCatchesWhatItCanSee(t *testing.T) {
	g := &gen{r: rand.New(rand.NewSource(20260815))}

	const n = 4000
	var broken, caught int
	missedBy := map[string]int{}

	for i := 0; i < n; i++ {
		src := g.program()
		prog, perr := parser.Parse(src)
		if perr != nil {
			t.Fatal(perr)
		}
		diags := checker.Check(prog)
		err := runs(src)
		if err == nil {
			continue
		}
		broken++
		if len(diags) > 0 {
			caught++
		} else {
			// Record the kind of the first line, so a miss is attributable to an
			// operation rather than being an anonymous count.
			key := strings.SplitN(strings.TrimSpace(strings.Split(src, "\n")[len(strings.Split(src, "\n"))-2]), "(", 2)[0]
			missedBy[key]++
		}
	}

	if broken == 0 {
		t.Fatal("the generator produced no broken programs, so this measures nothing")
	}
	pct := 100 * float64(caught) / float64(broken)
	t.Logf("of %d statically decidable shape errors, the checker caught %d (%.1f%%)", broken, caught, pct)
	for k, v := range missedBy {
		t.Logf("  missed, by leading form: %-40s %d", k, v)
	}
	if pct < 95 {
		t.Errorf("the checker caught %.1f%% of decidable shape errors, below the 95%% floor this test holds", pct)
	}
}
