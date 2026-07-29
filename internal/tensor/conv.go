package tensor

import "fmt"

// Conv2D computes a 2-D cross-correlation — the operation ML calls
// "convolution". input is [Cin, H, W], weight is [Cout, Cin, KH, KW], and the
// output is [Cout, OH, OW] with OH = H-KH+1, OW = W-KW+1 (valid padding, unit
// stride). It is differentiable in both the input and the weight.
func Conv2D(input, weight *Tensor) (*Tensor, error) {
	if len(input.Shape) != 3 {
		return nil, fmt.Errorf("conv2d: input must be [channels, height, width], got %v", input.Shape)
	}
	if len(weight.Shape) != 4 {
		return nil, fmt.Errorf("conv2d: weight must be [out, in, kh, kw], got %v", weight.Shape)
	}
	cin, h, w := input.Shape[0], input.Shape[1], input.Shape[2]
	cout, wcin, kh, kw := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	if wcin != cin {
		return nil, fmt.Errorf("conv2d: input has %d channels but weight expects %d", cin, wcin)
	}
	if kh > h || kw > w {
		return nil, fmt.Errorf("conv2d: kernel %dx%d is larger than input %dx%d", kh, kw, h, w)
	}
	oh, ow := h-kh+1, w-kw+1
	hw, khw, cinkhw := h*w, kh*kw, cin*kh*kw
	ohw := oh * ow
	out := make([]float64, cout*ohw)
	for co := 0; co < cout; co++ {
		for i := 0; i < oh; i++ {
			for j := 0; j < ow; j++ {
				var sum float64
				for ci := 0; ci < cin; ci++ {
					for a := 0; a < kh; a++ {
						for b := 0; b < kw; b++ {
							sum += input.Data[ci*hw+(i+a)*w+(j+b)] * weight.Data[co*cinkhw+ci*khw+a*kw+b]
						}
					}
				}
				out[co*ohw+i*ow+j] = sum
			}
		}
	}
	res := &Tensor{Data: out, Shape: []int{cout, oh, ow}}
	return track2(res, input, weight, func() {
		g := res.Grad // [Cout, OH, OW]
		var gi, gw []float64
		if input.RequiresGrad {
			gi = input.ensureGrad()
		}
		if weight.RequiresGrad {
			gw = weight.ensureGrad()
		}
		for co := 0; co < cout; co++ {
			for i := 0; i < oh; i++ {
				for j := 0; j < ow; j++ {
					gv := g[co*ohw+i*ow+j]
					if gv == 0 {
						continue
					}
					for ci := 0; ci < cin; ci++ {
						for a := 0; a < kh; a++ {
							for b := 0; b < kw; b++ {
								inIdx := ci*hw + (i+a)*w + (j + b)
								wIdx := co*cinkhw + ci*khw + a*kw + b
								if gi != nil {
									gi[inIdx] += gv * weight.Data[wIdx]
								}
								if gw != nil {
									gw[wIdx] += gv * input.Data[inIdx]
								}
							}
						}
					}
				}
			}
		}
	}), nil
}

// MaxPool2D applies non-overlapping k×k max pooling to each channel of a
// [C, H, W] tensor, producing [C, H/k, W/k] (a ragged remainder along either
// spatial axis is dropped). The gradient flows only to each window's max.
func MaxPool2D(input *Tensor, k int) (*Tensor, error) {
	if len(input.Shape) != 3 {
		return nil, fmt.Errorf("maxpool2d: input must be [channels, height, width], got %v", input.Shape)
	}
	if k < 1 {
		return nil, fmt.Errorf("maxpool2d: window must be >= 1, got %d", k)
	}
	c, h, w := input.Shape[0], input.Shape[1], input.Shape[2]
	oh, ow := h/k, w/k
	if oh == 0 || ow == 0 {
		return nil, fmt.Errorf("maxpool2d: window %d is larger than input %dx%d", k, h, w)
	}
	hw, ohw := h*w, oh*ow
	out := make([]float64, c*ohw)
	argmax := make([]int, c*ohw) // flat input index of each window's max, for backward
	for ch := 0; ch < c; ch++ {
		for i := 0; i < oh; i++ {
			for j := 0; j < ow; j++ {
				bestIdx := ch*hw + (i*k)*w + (j * k)
				best := input.Data[bestIdx]
				for a := 0; a < k; a++ {
					for b := 0; b < k; b++ {
						idx := ch*hw + (i*k+a)*w + (j*k + b)
						if input.Data[idx] > best {
							best = input.Data[idx]
							bestIdx = idx
						}
					}
				}
				out[ch*ohw+i*ow+j] = best
				argmax[ch*ohw+i*ow+j] = bestIdx
			}
		}
	}
	res := &Tensor{Data: out, Shape: []int{c, oh, ow}}
	return track1(res, input, func() {
		gi := input.ensureGrad()
		g := res.Grad
		for o := range g {
			gi[argmax[o]] += g[o]
		}
	}), nil
}
