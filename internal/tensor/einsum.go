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
func einsumRaw(inSubs []string, outSub string, inputs []*Tensor) (*Tensor, error) {
	for k, sub := range inSubs {
		if len(sub) != len(inputs[k].Shape) {
			return nil, fmt.Errorf("einsum: operand %d has rank %d but subscript %q has %d labels", k, len(inputs[k].Shape), sub, len(sub))
		}
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
	for step := 0; step < total; step++ {
		prod := 1.0
		for k := range inputs {
			prod *= inputs[k].Data[offK[k]]
		}
		outData[offOut] += prod
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
	return &Tensor{Data: outData, Shape: outShape}, nil
}

// Einsum evaluates an Einstein-summation spec over the inputs, with autodiff.
// The gradient of a (multilinear) einsum is itself an einsum, so backprop reuses
// the same machinery.
func Einsum(spec string, inputs []*Tensor) (*Tensor, error) {
	inSubs, outSub, err := ParseEinsum(spec, len(inputs))
	if err != nil {
		return nil, err
	}
	res, err := einsumRaw(inSubs, outSub, inputs)
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
			gp, err := einsumRaw(gSubs, inSubs[p], gInputs)
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
