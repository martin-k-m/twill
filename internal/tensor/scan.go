package tensor

import "math"

// Cumulative scans. Each runs over the tensor's elements in flat (row-major)
// order and keeps its shape. Unlike an elementwise op, output j depends on
// every input at or before j, so the backward pass walks the chain in reverse
// instead of pairing element with element.

// CumSum returns the running sum. Every d(out_j)/d(in_i) with i <= j is 1, so
// the gradient of in_i is the sum of the incoming gradients from i onward, i.e.
// a running sum taken from the end.
func CumSum(t *Tensor) *Tensor {
	n := len(t.Data)
	out := make([]float64, n)
	run := 0.0
	for i, v := range t.Data {
		run += v
		out[i] = run
	}
	res := &Tensor{Data: out, Shape: append([]int(nil), t.Shape...)}
	if recordJets && t.RequiresGrad {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			d, dd := 0.0, 0.0
			for i := 0; i < n; i++ {
				d += t.jet.d[i]
				dd += t.jet.dd[i]
				res.jet.d[i], res.jet.dd[i] = d, dd
			}
		}
	}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		run := 0.0
		for i := n - 1; i >= 0; i-- {
			run += res.Grad[i]
			gt[i] += run
		}
	})
}

// CumProd returns the running product. d(out_j)/d(in_i) for i <= j is out_j
// with in_i left out of it. Dividing out_j by in_i is the short way to say
// that and it breaks on a zero input, so the backward pass instead pairs a
// forward prefix product P_i = prod_{k<i} in_k with a backward accumulation
// S_i = g_i + in_{i+1}*S_{i+1}; the gradient of in_i is P_i * S_i.
func CumProd(t *Tensor) *Tensor {
	n := len(t.Data)
	out := make([]float64, n)
	run := 1.0
	for i, v := range t.Data {
		run *= v
		out[i] = run
	}
	res := &Tensor{Data: out, Shape: append([]int(nil), t.Shape...)}
	if recordJets && t.RequiresGrad {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			// out_j = out_{j-1} * in_j, differentiated by the product rule.
			prev, dPrev, ddPrev := 1.0, 0.0, 0.0
			for i := 0; i < n; i++ {
				x, dx, ddx := t.Data[i], t.jet.d[i], t.jet.dd[i]
				d := dPrev*x + prev*dx
				dd := ddPrev*x + 2*dPrev*dx + prev*ddx
				res.jet.d[i], res.jet.dd[i] = d, dd
				prev, dPrev, ddPrev = out[i], d, dd
			}
		}
	}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		suffix := make([]float64, n)
		s := 0.0
		for i := n - 1; i >= 0; i-- {
			if i+1 < n {
				s *= t.Data[i+1]
			}
			s += res.Grad[i]
			suffix[i] = s
		}
		p := 1.0
		for i := 0; i < n; i++ {
			gt[i] += p * suffix[i]
			p *= t.Data[i]
		}
	})
}

// cumExtreme implements the running max/min. Each output is one of the inputs
// (the best seen so far), so the gradient goes straight to that element and
// nowhere else. Ties keep the earlier element, the convention max/min/argmax
// already use.
func cumExtreme(t *Tensor, wantMax bool) *Tensor {
	n := len(t.Data)
	out := make([]float64, n)
	// arg[i] is the input index the running extreme came from, i.e. where the
	// gradient of out[i] belongs.
	arg := make([]int, n)
	for i, v := range t.Data {
		if i == 0 {
			out[i], arg[i] = v, 0
			continue
		}
		prev := out[i-1]
		if wantMax {
			out[i] = math.Max(prev, v)
		} else {
			out[i] = math.Min(prev, v)
		}
		if (wantMax && v > prev) || (!wantMax && v < prev) {
			arg[i] = i
		} else {
			arg[i] = arg[i-1]
		}
	}
	res := &Tensor{Data: out, Shape: append([]int(nil), t.Shape...)}
	if recordJets && t.RequiresGrad {
		res.jet = &jetState{}
		res.jet.jvp = func() {
			for i := 0; i < n; i++ {
				res.jet.d[i] = t.jet.d[arg[i]]
				res.jet.dd[i] = t.jet.dd[arg[i]]
			}
		}
	}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for i := 0; i < n; i++ {
			gt[arg[i]] += res.Grad[i]
		}
	})
}

func CumMax(t *Tensor) *Tensor { return cumExtreme(t, true) }
func CumMin(t *Tensor) *Tensor { return cumExtreme(t, false) }

// --- running totals --------------------------------------------------------

// CumsumAxis is the running sum along one axis. The shape is unchanged, since
// every element gets an output rather than the axis collapsing.
func CumsumAxis(t *Tensor, axis int) (*Tensor, error) {
	axis, err := normalizeAxis(axis, len(t.Shape))
	if err != nil {
		return nil, err
	}
	before, L, after := axisSpans(t.Shape, axis)

	out := make([]float64, len(t.Data))
	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			base := i*L*after + j
			run := 0.0
			for k := 0; k < L; k++ {
				run += t.Data[base+k*after]
				out[base+k*after] = run
			}
		}
	}

	res := &Tensor{Data: out, Shape: append([]int{}, t.Shape...)}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		// Element k feeds every output from k onwards, so its gradient is the
		// sum of those outputs' gradients. That is the running sum again, taken
		// backwards, which is why this is a second pass and not a nested loop.
		for i := 0; i < before; i++ {
			for j := 0; j < after; j++ {
				base := i*L*after + j
				run := 0.0
				for k := L - 1; k >= 0; k-- {
					run += res.Grad[base+k*after]
					gt[base+k*after] += run
				}
			}
		}
	}), nil
}

// CumprodAxis is the running product along one axis, with the shape unchanged.
func CumprodAxis(t *Tensor, axis int) (*Tensor, error) {
	axis, err := normalizeAxis(axis, len(t.Shape))
	if err != nil {
		return nil, err
	}
	before, L, after := axisSpans(t.Shape, axis)

	out := make([]float64, len(t.Data))
	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			base := i*L*after + j
			run := 1.0
			for k := 0; k < L; k++ {
				run *= t.Data[base+k*after]
				out[base+k*after] = run
			}
		}
	}

	res := &Tensor{Data: out, Shape: append([]int{}, t.Shape...)}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		// out[n] is the product of x[0..n], so d out[n] / d x[k] is the product
		// of that run with x[k] left out, for every n at or after k.
		//
		// The tempting shortcut is out[n]/x[k], and it is wrong exactly when
		// x[k] is zero, which is not a rare value for a tensor. Rebuilding the
		// product instead costs a factor of L and is always right. The runs are
		// one axis long, not the whole tensor, so this stays cheap in the shapes
		// anybody actually writes.
		for i := 0; i < before; i++ {
			for j := 0; j < after; j++ {
				base := i*L*after + j
				for k := 0; k < L; k++ {
					total := 0.0
					// The product of x[0..n] without x[k], built forwards so it
					// carries over from one n to the next.
					without := 1.0
					for m := 0; m < k; m++ {
						without *= t.Data[base+m*after]
					}
					for n := k; n < L; n++ {
						if n > k {
							without *= t.Data[base+n*after]
						}
						total += res.Grad[base+n*after] * without
					}
					gt[base+k*after] += total
				}
			}
		}
	}), nil
}

// --- axis-aware scans ------------------------------------------------------
//
// The scans above run over the tensor's elements in order, which is what a
// sequence wants and is all a 1-D tensor can mean. On anything with more than
// one axis a caller usually means one scan per row, so these take the axis and
// keep the shape.

// CumMaxAxis is the running maximum along one axis, with the shape unchanged.
func CumMaxAxis(t *Tensor, axis int) (*Tensor, error) { return cumExtremeAxis(t, axis, true) }

// CumMinAxis is the running minimum along one axis, with the shape unchanged.
func CumMinAxis(t *Tensor, axis int) (*Tensor, error) { return cumExtremeAxis(t, axis, false) }

func cumExtremeAxis(t *Tensor, axis int, wantMax bool) (*Tensor, error) {
	axis, err := normalizeAxis(axis, len(t.Shape))
	if err != nil {
		return nil, err
	}
	before, L, after := axisSpans(t.Shape, axis)

	out := make([]float64, len(t.Data))
	// arg[p] is the input position the running extreme at p came from, which is
	// where the gradient of that output belongs.
	arg := make([]int, len(t.Data))

	for i := 0; i < before; i++ {
		for j := 0; j < after; j++ {
			base := i*L*after + j
			for k := 0; k < L; k++ {
				p := base + k*after
				if k == 0 {
					out[p], arg[p] = t.Data[p], p
					continue
				}
				prev := out[p-after]
				v := t.Data[p]
				// A tie keeps the earlier element, matching the flat scan and
				// matching sort's stability: whichever rule is chosen has to be
				// the same one everywhere, or a gradient moves when a value is
				// merely repeated.
				if (wantMax && v > prev) || (!wantMax && v < prev) {
					out[p], arg[p] = v, p
				} else {
					out[p], arg[p] = prev, arg[p-after]
				}
			}
		}
	}

	res := &Tensor{Data: out, Shape: append([]int{}, t.Shape...)}
	return track1(res, t, func() {
		if !t.RequiresGrad {
			return
		}
		gt := t.ensureGrad()
		for p := range out {
			gt[arg[p]] += res.Grad[p]
		}
	}), nil
}
