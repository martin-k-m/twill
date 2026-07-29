package gbm

import (
	"math"
	"math/rand"
	"testing"
)

// makeData builds n rows of d features from a fixed RNG, plus a target from f.
func makeData(rng *rand.Rand, n, d int, f func(x []float64) float64) (X, y []float64) {
	X = make([]float64, n*d)
	y = make([]float64, n)
	row := make([]float64, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			v := rng.Float64()
			X[i*d+j] = v
			row[j] = v
		}
		y[i] = f(row)
	}
	return X, y
}

func rmse(pred, y []float64) float64 {
	var s float64
	for i := range y {
		e := pred[i] - y[i]
		s += e * e
	}
	return math.Sqrt(s / float64(len(y)))
}

func TestRegressionFitsTrainingData(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	n, d := 300, 4
	target := func(x []float64) float64 {
		v := 1.5*x[0] - 2.0*x[1]
		if x[2] > 0.5 {
			v += 1.0
		}
		return v
	}
	X, y := makeData(rng, n, d, target)

	p := DefaultParams()
	p.Rounds = 200
	p.MaxDepth = 4
	p.LearningRate = 0.1
	m, err := Fit(X, y, n, d, p)
	if err != nil {
		t.Fatal(err)
	}
	pred, err := m.Predict(X, n, d)
	if err != nil {
		t.Fatal(err)
	}
	if got := rmse(pred, y); got > 0.1 {
		t.Fatalf("training RMSE too high: %g", got)
	}
}

func TestLogisticSeparates(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	n, d := 400, 3
	target := func(x []float64) float64 {
		if x[0]+x[1] > 1.0 {
			return 1
		}
		return 0
	}
	X, y := makeData(rng, n, d, target)

	p := DefaultParams()
	p.Rounds = 150
	p.MaxDepth = 3
	p.Objective = Logistic
	m, err := Fit(X, y, n, d, p)
	if err != nil {
		t.Fatal(err)
	}
	pred, err := m.Predict(X, n, d)
	if err != nil {
		t.Fatal(err)
	}
	correct := 0
	for i := range y {
		if pred[i] < 0 || pred[i] > 1 {
			t.Fatalf("logistic prediction %g out of [0,1]", pred[i])
		}
		phat := 0.0
		if pred[i] >= 0.5 {
			phat = 1
		}
		if phat == y[i] {
			correct++
		}
	}
	if acc := float64(correct) / float64(n); acc < 0.95 {
		t.Fatalf("training accuracy too low: %g", acc)
	}
}

// TestDeterministic checks that two fits on identical inputs — and thus any two
// runs regardless of how the parallel split search is scheduled — produce
// bit-identical predictions.
func TestDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	n, d := 2000, 6 // large enough to exercise the parallel split path
	target := func(x []float64) float64 { return x[0]*x[0] - x[3] + 0.5*x[5] }
	X, y := makeData(rng, n, d, target)

	p := DefaultParams()
	p.Rounds = 40
	p.MaxDepth = 5

	m1, err := Fit(X, y, n, d, p)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := Fit(X, y, n, d, p)
	if err != nil {
		t.Fatal(err)
	}
	pred1, _ := m1.Predict(X, n, d)
	pred2, _ := m2.Predict(X, n, d)
	for i := range pred1 {
		if pred1[i] != pred2[i] {
			t.Fatalf("non-deterministic prediction at row %d: %v != %v", i, pred1[i], pred2[i])
		}
	}
}

func TestRejectsBadLabels(t *testing.T) {
	X := []float64{0, 1, 2, 3}
	y := []float64{0, 2} // 2 is not a valid logistic label
	p := DefaultParams()
	p.Objective = Logistic
	if _, err := Fit(X, y, 2, 2, p); err == nil {
		t.Fatal("expected an error for non-binary logistic labels")
	}
}
