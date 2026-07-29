// Package gbm implements gradient-boosted regression trees in pure Go, with no
// dependencies. It uses the second-order (Newton) formulation — the same one
// XGBoost/LightGBM use — so both squared-error regression and logistic (binary)
// classification fall out of the same tree builder by swapping the per-sample
// gradient and hessian.
//
// The engine is deterministic: given the same data and parameters it produces
// bit-identical trees, regardless of how the per-feature split search is spread
// across CPU cores. That reproducibility is the point for finance use.
package gbm

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
)

// Objective selects the loss the boosting minimizes.
type Objective int

const (
	// Squared is 1/2 (y - F)^2: plain regression. Predictions are the raw score.
	Squared Objective = iota
	// Logistic is the binary log-loss on labels in {0, 1}. Predictions are the
	// probability sigmoid(F).
	Logistic
)

// Params configures a fit. Zero values are not valid defaults; use
// DefaultParams and override.
type Params struct {
	Rounds       int       // number of boosting rounds (trees)
	LearningRate float64   // shrinkage applied to each tree's contribution
	MaxDepth     int       // maximum tree depth (0 = stumps of a single leaf)
	MinLeaf      int       // minimum samples in a leaf
	Lambda       float64   // L2 regularization on leaf weights
	Gamma        float64   // minimum gain required to make a split
	Objective    Objective // loss function
}

// DefaultParams returns sensible starting parameters for tabular data.
func DefaultParams() Params {
	return Params{
		Rounds:       100,
		LearningRate: 0.1,
		MaxDepth:     3,
		MinLeaf:      1,
		Lambda:       1.0,
		Gamma:        0.0,
		Objective:    Squared,
	}
}

// tree is one boosted regression tree, flattened into parallel arrays indexed
// by node id. Node 0 is the root. For an internal node, feature >= 0 and a
// sample goes left when x[feature] <= threshold; for a leaf, feature == -1 and
// leaf holds the node's output weight.
type tree struct {
	feature   []int
	threshold []float64
	left      []int
	right     []int
	leaf      []float64
}

// Model is a trained ensemble. It is returned as an opaque runtime value.
type Model struct {
	Base      float64 // initial score every prediction starts from
	LR        float64 // learning rate baked into prediction
	NFeat     int     // number of input features expected
	Objective Objective
	trees     []tree
}

// String renders a short summary (used when a model is printed).
func (m *Model) String() string {
	obj := "squared"
	if m.Objective == Logistic {
		obj = "logistic"
	}
	return fmt.Sprintf("<gbm %s, %d trees, %d features>", obj, len(m.trees), m.NFeat)
}

// Fit trains a model on X (n rows by d features, row-major) and targets y
// (length n).
func Fit(X, y []float64, n, d int, p Params) (*Model, error) {
	if n <= 0 || d <= 0 {
		return nil, fmt.Errorf("gbm: need at least one row and one feature (got %d x %d)", n, d)
	}
	if len(X) != n*d {
		return nil, fmt.Errorf("gbm: X has %d values but n*d = %d", len(X), n*d)
	}
	if len(y) != n {
		return nil, fmt.Errorf("gbm: y has %d values but n = %d", len(y), n)
	}
	if p.Rounds < 0 {
		return nil, fmt.Errorf("gbm: rounds must be >= 0")
	}
	if p.Lambda < 0 || p.Gamma < 0 {
		return nil, fmt.Errorf("gbm: lambda and gamma must be >= 0")
	}
	if p.Objective == Logistic {
		for i, v := range y {
			if v != 0 && v != 1 {
				return nil, fmt.Errorf("gbm: logistic labels must be 0 or 1 (row %d is %g)", i, v)
			}
		}
	}
	minLeaf := p.MinLeaf
	if minLeaf < 1 {
		minLeaf = 1
	}

	base := baseScore(y, p.Objective)
	F := make([]float64, n)
	for i := range F {
		F[i] = base
	}

	// Pre-sort each feature's row indices once; children reuse the order by
	// partitioning it, so no node ever re-sorts.
	order := make([][]int32, d)
	for f := 0; f < d; f++ {
		idx := make([]int32, n)
		for i := range idx {
			idx[i] = int32(i)
		}
		ff := f
		sort.SliceStable(idx, func(a, b int) bool {
			return X[int(idx[a])*d+ff] < X[int(idx[b])*d+ff]
		})
		order[f] = idx
	}

	g := make([]float64, n)
	h := make([]float64, n)
	m := &Model{Base: base, LR: p.LearningRate, NFeat: d, Objective: p.Objective}
	bld := &builder{X: X, d: d, minLeaf: minLeaf, lambda: p.Lambda, gamma: p.Gamma, maxDepth: p.MaxDepth, side: make([]bool, n)}

	for r := 0; r < p.Rounds; r++ {
		gradients(y, F, g, h, p.Objective)
		bld.g, bld.h = g, h
		t := bld.build(order)
		// Fold the tree's contribution into the running score.
		for i := 0; i < n; i++ {
			F[i] += p.LearningRate * predictRow(t, X, i*d)
		}
		m.trees = append(m.trees, t)
	}
	return m, nil
}

// Predict scores X (n rows by d features, row-major) and returns one value per
// row: the raw score for Squared, or a probability for Logistic.
func (m *Model) Predict(X []float64, n, d int) ([]float64, error) {
	if d != m.NFeat {
		return nil, fmt.Errorf("gbm: model expects %d features but X has %d", m.NFeat, d)
	}
	if len(X) != n*d {
		return nil, fmt.Errorf("gbm: X has %d values but n*d = %d", len(X), n*d)
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		s := m.Base
		base := i * d
		for ti := range m.trees {
			s += m.LR * predictRow(m.trees[ti], X, base)
		}
		if m.Objective == Logistic {
			s = sigmoid(s)
		}
		out[i] = s
	}
	return out, nil
}

func baseScore(y []float64, obj Objective) float64 {
	var sum float64
	for _, v := range y {
		sum += v
	}
	mu := sum / float64(len(y))
	if obj == Logistic {
		mu = clamp(mu, 1e-6, 1-1e-6)
		return math.Log(mu / (1 - mu))
	}
	return mu
}

func gradients(y, F, g, h []float64, obj Objective) {
	switch obj {
	case Logistic:
		for i := range y {
			p := sigmoid(F[i])
			g[i] = p - y[i]
			h[i] = p * (1 - p)
		}
	default: // Squared
		for i := range y {
			g[i] = F[i] - y[i]
			h[i] = 1
		}
	}
}

func predictRow(t tree, X []float64, base int) float64 {
	node := 0
	for t.feature[node] >= 0 {
		if X[base+t.feature[node]] <= t.threshold[node] {
			node = t.left[node]
		} else {
			node = t.right[node]
		}
	}
	return t.leaf[node]
}

// builder holds the per-fit state for growing one tree at a time.
type builder struct {
	X        []float64
	g, h     []float64
	d        int
	minLeaf  int
	lambda   float64
	gamma    float64
	maxDepth int
	side     []bool // scratch: true = row goes left, indexed by global row id
	t        tree
}

// build grows a tree over the given per-feature sorted orders and returns it.
func (bld *builder) build(order [][]int32) tree {
	bld.t = tree{}
	// The root owns a private copy of each feature order, since growth mutates
	// (partitions) the orders it is handed.
	root := make([][]int32, bld.d)
	for f := range order {
		root[f] = append([]int32(nil), order[f]...)
	}
	bld.grow(root, 0)
	return bld.t
}

// grow builds the subtree for the rows described by nodeOrder and returns its
// node id. nodeOrder[f] lists this node's rows sorted ascending by feature f.
func (bld *builder) grow(nodeOrder [][]int32, depth int) int {
	rows := nodeOrder[0]
	var G, H float64
	for _, idx := range rows {
		G += bld.g[idx]
		H += bld.h[idx]
	}
	if depth >= bld.maxDepth || len(rows) < 2*bld.minLeaf {
		return bld.newLeaf(G, H)
	}
	best := bld.bestSplit(nodeOrder, G, H)
	if !best.valid || best.gain <= 0 {
		return bld.newLeaf(G, H)
	}

	// Record which side each of this node's rows falls on, then split every
	// feature's order accordingly so the children stay sorted.
	of := nodeOrder[best.feature]
	for k, idx := range of {
		bld.side[idx] = k < best.leftCount
	}
	left, right := partition(nodeOrder, bld.side)

	id := bld.newInternal(best.feature, best.threshold)
	l := bld.grow(left, depth+1)
	r := bld.grow(right, depth+1)
	bld.t.left[id] = l
	bld.t.right[id] = r
	return id
}

type splitResult struct {
	valid     bool
	gain      float64
	feature   int
	threshold float64
	leftCount int
}

// bestSplit searches every feature for the split with the highest gain,
// evaluating features in parallel for large nodes. The reduction is done in
// fixed feature order, so the result never depends on goroutine scheduling.
func (bld *builder) bestSplit(nodeOrder [][]int32, G, H float64) splitResult {
	d := bld.d
	results := make([]splitResult, d)
	rows := len(nodeOrder[0])

	if d > 1 && rows >= 1024 {
		workers := runtime.GOMAXPROCS(0)
		if workers > d {
			workers = d
		}
		var next int32 = -1
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					f := int(atomic.AddInt32(&next, 1))
					if f >= d {
						return
					}
					results[f] = bld.featureSplit(nodeOrder[f], f, G, H)
				}
			}()
		}
		wg.Wait()
	} else {
		for f := 0; f < d; f++ {
			results[f] = bld.featureSplit(nodeOrder[f], f, G, H)
		}
	}

	best := splitResult{}
	for f := 0; f < d; f++ {
		if results[f].valid && (!best.valid || results[f].gain > best.gain) {
			best = results[f]
		}
	}
	return best
}

// featureSplit finds the best split on a single feature by scanning its sorted
// rows and accumulating the left-side gradient/hessian sums.
func (bld *builder) featureSplit(of []int32, f int, G, H float64) splitResult {
	rows := len(of)
	parent := G * G / (H + bld.lambda)
	var GL, HL float64
	best := splitResult{}
	for k := 0; k < rows-1; k++ {
		idx := of[k]
		GL += bld.g[idx]
		HL += bld.h[idx]
		v := bld.X[int(idx)*bld.d+f]
		vNext := bld.X[int(of[k+1])*bld.d+f]
		if v == vNext {
			continue // cannot place a boundary between equal values
		}
		leftCount := k + 1
		if leftCount < bld.minLeaf || rows-leftCount < bld.minLeaf {
			continue
		}
		GR, HR := G-GL, H-HL
		gain := 0.5*(GL*GL/(HL+bld.lambda)+GR*GR/(HR+bld.lambda)-parent) - bld.gamma
		if !best.valid || gain > best.gain {
			best = splitResult{valid: true, gain: gain, feature: f, threshold: (v + vNext) / 2, leftCount: leftCount}
		}
	}
	return best
}

// partition splits each feature's sorted order into the left and right child
// orders, preserving sortedness, according to side[rowID].
func partition(nodeOrder [][]int32, side []bool) (left, right [][]int32) {
	d := len(nodeOrder)
	left = make([][]int32, d)
	right = make([][]int32, d)
	for f := 0; f < d; f++ {
		of := nodeOrder[f]
		nl := 0
		for _, idx := range of {
			if side[idx] {
				nl++
			}
		}
		l := make([]int32, 0, nl)
		r := make([]int32, 0, len(of)-nl)
		for _, idx := range of {
			if side[idx] {
				l = append(l, idx)
			} else {
				r = append(r, idx)
			}
		}
		left[f] = l
		right[f] = r
	}
	return left, right
}

func (bld *builder) newLeaf(G, H float64) int {
	w := -G / (H + bld.lambda)
	id := len(bld.t.feature)
	bld.t.feature = append(bld.t.feature, -1)
	bld.t.threshold = append(bld.t.threshold, 0)
	bld.t.left = append(bld.t.left, -1)
	bld.t.right = append(bld.t.right, -1)
	bld.t.leaf = append(bld.t.leaf, w)
	return id
}

func (bld *builder) newInternal(feat int, thr float64) int {
	id := len(bld.t.feature)
	bld.t.feature = append(bld.t.feature, feat)
	bld.t.threshold = append(bld.t.threshold, thr)
	bld.t.left = append(bld.t.left, -1)
	bld.t.right = append(bld.t.right, -1)
	bld.t.leaf = append(bld.t.leaf, 0)
	return id
}

func sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}
