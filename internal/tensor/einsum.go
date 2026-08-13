package tensor

import (
	"fmt"
	"sort"
	"strings"
)

// ParseEinsum parses an einsum spec like "ij,jk->ik" for n inputs. Without an
// explicit "->", the output is the labels that appear exactly once, in sorted
// order (the NumPy implicit convention). Repeated labels within a single input
// subscript (traces/diagonals) are not supported.
func ParseEinsum(spec string, n int) (inSubs []string, outSub string, err error) {
	spec = strings.ReplaceAll(spec, " ", "")
	var inPart string
	explicit := false
	if i := strings.Index(spec, "->"); i >= 0 {
		inPart = spec[:i]
		outSub = spec[i+2:]
		explicit = true
	} else {
		inPart = spec
	}
	inSubs = strings.Split(inPart, ",")
	if len(inSubs) != n {
		return nil, "", fmt.Errorf("einsum: spec has %d operands but got %d", len(inSubs), n)
	}
	counts := map[rune]int{}
	for _, sub := range inSubs {
		seen := map[rune]bool{}
		for _, ch := range sub {
			if ch < 'a' || ch > 'z' {
				return nil, "", fmt.Errorf("einsum: labels must be lowercase letters, found %q", string(ch))
			}
			if seen[ch] {
				return nil, "", fmt.Errorf("einsum: repeated label %q within one operand is not supported", string(ch))
			}
			seen[ch] = true
			counts[ch]++
		}
	}
	if !explicit {
		var once []rune
		for ch, c := range counts {
			if c == 1 {
				once = append(once, ch)
			}
		}
		sort.Slice(once, func(i, j int) bool { return once[i] < once[j] })
		outSub = string(once)
	}
	// Validate the output subscript.
	oseen := map[rune]bool{}
	for _, ch := range outSub {
		if counts[ch] == 0 {
			return nil, "", fmt.Errorf("einsum: output label %q does not appear in the inputs", string(ch))
		}
		if oseen[ch] {
			return nil, "", fmt.Errorf("einsum: repeated output label %q", string(ch))
		}
		oseen[ch] = true
	}
	return inSubs, outSub, nil
}

// EinsumOutputDims resolves the output shape from input shapes (dimensions may
// be -1 for unknown). ok is false only when the spec itself is invalid.
func EinsumOutputDims(inSubs []string, outSub string, inputDims [][]int) (dims []int, err error) {
	labelDim := map[rune]int{}
	for k, sub := range inSubs {
		if len(sub) != len(inputDims[k]) {
			return nil, fmt.Errorf("einsum: operand %d has rank %d but subscript %q has %d labels", k, len(inputDims[k]), sub, len(sub))
		}
		for a, ch := range sub {
			d := inputDims[k][a]
			if prev, ok := labelDim[ch]; ok {
				if prev >= 0 && d >= 0 && prev != d {
					return nil, fmt.Errorf("einsum: label %q has inconsistent sizes %d and %d", string(ch), prev, d)
				}
				if prev < 0 {
					labelDim[ch] = d
				}
			} else {
				labelDim[ch] = d
			}
		}
	}
	out := make([]int, 0, len(outSub))
	for _, ch := range outSub {
		out = append(out, labelDim[ch])
	}
	return out, nil
}

// einsumRaw computes the einsum forward pass without autodiff.
func einsumRaw(inSubs []string, outSub string, inputs []*Tensor, dt DType) (*Tensor, error) {
	for k, sub := range inSubs {
		if len(sub) != len(inputs[k].Shape) {
			return nil, fmt.Errorf("einsum: operand %d has rank %d but subscript %q has %d labels", k, len(inputs[k].Shape), sub, len(sub))
		}
	}
	// Fast path: the rank-3 batched matmul that attention spells as an einsum.
	// The general odometer below re-derives a multi-dimensional index for every
	// scalar multiply; dispatching each batch to the tuned 2-D kernels removes
	// that per-element bookkeeping and inherits their tiling and accumulators.
	if r := tryBatchedMatmul(inSubs, outSub, inputs, dt); r != nil {
		return r, nil
	}
	inputDims := make([][]int, len(inputs))
	for k := range inputs {
		inputDims[k] = inputs[k].Shape
	}
	outShape, err := EinsumOutputDims(inSubs, outSub, inputDims)
	if err != nil {
		return nil, err
	}

	// Collect the distinct labels across all inputs, with their sizes.
	labelDim := map[rune]int{}
	var labels []rune
	for k, sub := range inSubs {
		for a, ch := range sub {
			if _, ok := labelDim[ch]; !ok {
				labelDim[ch] = inputs[k].Shape[a]
				labels = append(labels, ch)
			}
		}
	}
	L := len(labels)
	dims := make([]int, L)
	for i, ch := range labels {
		dims[i] = labelDim[ch]
	}
	labelIndex := map[rune]int{}
	for i, ch := range labels {
		labelIndex[ch] = i
	}

	// Per label, the stride contribution into each input and into the output.
	inContrib := make([][]int, L)
	outContrib := make([]int, L)
	for li := range inContrib {
		inContrib[li] = make([]int, len(inputs))
	}
	for k := range inputs {
		st := strides(inputs[k].Shape)
		for a, ch := range inSubs[k] {
			inContrib[labelIndex[ch]][k] += st[a]
		}
	}
	outStr := strides(outShape)
	for a, ch := range outSub {
		outContrib[labelIndex[ch]] = outStr[a]
	}

	outData := make([]float64, numel(outShape))
	total := 1
	for _, d := range dims {
		total *= d
	}
	counter := make([]int, L)
	offK := make([]int, len(inputs))
	offOut := 0
	// A narrow contraction rounds each accumulation step to the accumulation
	// dtype; f64 is a plain running sum, bit-identical to before.
	acc := AccDType(dt)
	narrow := dt != DTF64
	for step := 0; step < total; step++ {
		prod := 1.0
		for k := range inputs {
			prod *= inputs[k].Data[offK[k]]
		}
		if narrow {
			outData[offOut] = RoundToDType(acc, outData[offOut]+prod)
		} else {
			outData[offOut] += prod
		}
		for li := L - 1; li >= 0; li-- {
			counter[li]++
			for k := range inputs {
				offK[k] += inContrib[li][k]
			}
			offOut += outContrib[li]
			if counter[li] < dims[li] {
				break
			}
			counter[li] = 0
			for k := range inputs {
				offK[k] -= inContrib[li][k] * dims[li]
			}
			offOut -= outContrib[li] * dims[li]
		}
	}
	return contractionResult(outData, outShape, dt), nil
}

// tryBatchedMatmul handles the two rank-3 batched matmuls attention is built
// from -- "bxc,byc->bxy" (contracting the last axis of each, e.g. Q·Kᵀ over
// heads) and "bxc,bcy->bxy" (a per-batch matmul, e.g. attn·V) -- by running each
// batch through the tuned 2-D kernels. It returns nil for any other spec, so the
// caller falls back to the general path; a shape it does not recognise is never
// mishandled, only left alone. The result differs from the odometer by the
// kernels' summation order, within tolerance, as the fused linear kernel does.
func tryBatchedMatmul(inSubs []string, outSub string, inputs []*Tensor, dt DType) *Tensor {
	if len(inputs) != 2 || len(outSub) != 3 {
		return nil
	}
	a, bSub := inSubs[0], inSubs[1]
	if len(a) != 3 || len(bSub) != 3 {
		return nil
	}
	// Batch label leads all three; the output's other two labels are the free
	// axes, one from each input. Require the first input to be [batch, x, c] with
	// x the output's first free axis and c the contracted axis (the shape
	// attention produces); anything else falls back.
	bl := outSub[0]
	x, y := outSub[1], outSub[2]
	if a[0] != bl || bSub[0] != bl || a[1] != x {
		return nil
	}
	c := a[2] // contracted: must not appear in the output
	if c == x || c == y {
		return nil
	}
	ta, tb := inputs[0], inputs[1]
	B := ta.Shape[0]
	if tb.Shape[0] != B {
		return nil
	}
	X, C := ta.Shape[1], ta.Shape[2]
	// Second input holds y and c in some order; that order picks the kernel.
	var Y int
	transposed := false // true when b is [batch, y, c] and we need y·cᵀ (mmNT)
	if bSub[1] == y && bSub[2] == c {
		transposed = true
		Y = tb.Shape[1]
		if tb.Shape[2] != C {
			return nil
		}
	} else if bSub[1] == c && bSub[2] == y {
		Y = tb.Shape[2]
		if tb.Shape[1] != C {
			return nil
		}
	} else {
		return nil
	}
	out := make([]float64, B*X*Y)
	narrow := dt != DTF64
	acc := AccDType(dt)
	for batch := 0; batch < B; batch++ {
		aMat := ta.Data[batch*X*C : batch*X*C+X*C]
		var oMat []float64
		if transposed {
			bMat := tb.Data[batch*Y*C : batch*Y*C+Y*C]
			if narrow {
				oMat = mmAccNT(aMat, X, C, bMat, Y, acc)
			} else {
				oMat = mmNT(aMat, X, C, bMat, Y) // [X,C] · [Y,C]ᵀ -> [X,Y]
			}
		} else {
			bMat := tb.Data[batch*C*Y : batch*C*Y+C*Y]
			if narrow {
				oMat = mmAcc(aMat, X, C, bMat, Y, acc)
			} else {
				oMat = mm(aMat, X, C, bMat, Y) // [X,C] · [C,Y] -> [X,Y]
			}
		}
		copy(out[batch*X*Y:batch*X*Y+X*Y], oMat)
	}
	return contractionResult(out, []int{B, X, Y}, dt)
}

// Einsum evaluates an Einstein-summation spec over the inputs, with autodiff.
// The gradient of a (multilinear) einsum is itself an einsum, so backprop reuses
// the same machinery.
func Einsum(spec string, inputs []*Tensor) (*Tensor, error) {
	inSubs, outSub, err := ParseEinsum(spec, len(inputs))
	if err != nil {
		return nil, err
	}
	// A contraction produces the promotion of its operands and accumulates in f32
	// for anything narrower (docs/dtypes.md). The gradient einsums below stay f64.
	dt := inputs[0].DType()
	for _, in := range inputs[1:] {
		dt = Promote(dt, in.DType())
	}
	res, err := einsumRaw(inSubs, outSub, inputs, dt)
	if err != nil {
		return nil, err
	}
	rg := false
	for _, in := range inputs {
		if in.RequiresGrad {
			rg = true
			break
		}
	}
	if !rg {
		return res, nil
	}
	prev := append([]*Tensor(nil), inputs...)
	return trackN(res, prev, func() {
		gradOut := &Tensor{Data: res.Grad, Shape: res.Shape}
		for p := range inputs {
			if !inputs[p].RequiresGrad {
				continue
			}
			// d/d(input_p) = einsum with input_p's subscript as the output and
			// the upstream gradient standing in for the original output.
			gInputs := []*Tensor{gradOut}
			gSubs := []string{outSub}
			for j := range inputs {
				if j != p {
					gInputs = append(gInputs, inputs[j])
					gSubs = append(gSubs, inSubs[j])
				}
			}
			gp, err := einsumRaw(gSubs, inSubs[p], gInputs, DTF64)
			if err != nil {
				continue
			}
			ga := inputs[p].ensureGrad()
			for i := range ga {
				ga[i] += gp.Data[i]
			}
		}
	}), nil
}
