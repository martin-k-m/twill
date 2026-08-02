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
