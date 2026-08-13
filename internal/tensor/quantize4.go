package tensor

import (
	"fmt"
	"math"
)

// QTensorI4 is a weight in 4-bit form: two signed nibbles per byte, with one f64
// scale per block of Block elements along each row. Four bits is too few to share
// one scale across a whole row — a single outlier would coarsen every other
// weight — so the scale is per block, the scheme quantised LLM runtimes use. The
// result is roughly 0.6 bytes per weight (a nibble plus the amortised block
// scale), about 13x smaller than f64 and half of int8.
type QTensorI4 struct {
	Q     []byte    // packed nibbles, 2 per byte, row-major within each row
	Scale []float64 // one per block, len = Rows * blocksPerRow
	Rows  int
	Cols  int
	Block int // elements per scale block
}

func blocksPerRow(cols, block int) int { return (cols + block - 1) / block }
func rowBytes(cols int) int            { return (cols + 1) / 2 }

// QuantizeI4 packs a 2-D weight [n, k] into 4-bit blocks. Each block's scale maps
// its largest magnitude to 7 (symmetric, so codes run -7..7 and -8 is unused),
// which keeps the packing invertible and the step uniform inside the block.
func QuantizeI4(w *Tensor, block int) (*QTensorI4, error) {
	if len(w.Shape) != 2 {
		return nil, fmt.Errorf("quantize expects a 2-D weight, got rank %d", len(w.Shape))
	}
	if block <= 0 {
		block = 32
	}
	n, k := w.Shape[0], w.Shape[1]
	bpr := blocksPerRow(k, block)
	rb := rowBytes(k)
	q := &QTensorI4{
		Q:     make([]byte, n*rb),
		Scale: make([]float64, n*bpr),
		Rows:  n, Cols: k, Block: block,
	}
	for j := 0; j < n; j++ {
		row := w.Data[j*k : j*k+k]
		for b := 0; b < bpr; b++ {
			p0 := b * block
			p1 := p0 + block
			if p1 > k {
				p1 = k
			}
			maxAbs := 0.0
			for p := p0; p < p1; p++ {
				if a := math.Abs(row[p]); a > maxAbs {
					maxAbs = a
				}
			}
			var s, inv float64
			if maxAbs > 0 {
				s = maxAbs / 7
				inv = 1 / s
			}
			q.Scale[j*bpr+b] = s
			for p := p0; p < p1; p++ {
				code := 0
				if inv != 0 {
					r := math.Round(row[p] * inv)
					if r > 7 {
						r = 7
					} else if r < -7 {
						r = -7
					}
					code = int(r)
				}
				setNibble(q.Q, j*rb, p, code)
			}
		}
	}
	return q, nil
}

// setNibble writes a signed 4-bit code (-8..7) at element p of a row whose bytes
// start at base. Two's-complement in four bits: a negative code is stored as
// code+16, so 0xF is -1.
func setNibble(buf []byte, base, p, code int) {
	nib := byte(code & 0x0F)
	idx := base + p/2
	if p&1 == 0 {
		buf[idx] = (buf[idx] & 0xF0) | nib
	} else {
		buf[idx] = (buf[idx] & 0x0F) | (nib << 4)
	}
}

// getNibble reads element p back as a signed int in -8..7.
func getNibble(buf []byte, base, p int) int {
	b := buf[base+p/2]
	var nib byte
	if p&1 == 0 {
		nib = b & 0x0F
	} else {
		nib = b >> 4
	}
	if nib >= 8 {
		return int(nib) - 16
	}
	return int(nib)
}

// Dequantize reconstructs the f64 weight, for tests and printing.
func (q *QTensorI4) Dequantize() *Tensor {
	d := make([]float64, q.Rows*q.Cols)
	bpr := blocksPerRow(q.Cols, q.Block)
	rb := rowBytes(q.Cols)
	for j := 0; j < q.Rows; j++ {
		for p := 0; p < q.Cols; p++ {
			s := q.Scale[j*bpr+p/q.Block]
			d[j*q.Cols+p] = float64(getNibble(q.Q, j*rb, p)) * s
		}
	}
	return &Tensor{Data: d, Shape: []int{q.Rows, q.Cols}}
}

// Bytes is the packed footprint: half a byte per weight plus eight per block
// scale. For the default block of 32 that is 0.5 + 8/32 = 0.75 bytes per weight.
func (q *QTensorI4) Bytes() int { return len(q.Q) + len(q.Scale)*8 }

// QLinear4 computes x @ Wᵀ where W is the 4-bit weight q. The dequant is applied
// per block: within a block the int4 codes accumulate against x, then the running
// sum is scaled once by the block's f64 scale, so decoding costs a nibble read
// and the scaling costs one multiply per block, not per element.
func QLinear4(x *Tensor, q *QTensorI4) (*Tensor, error) {
	x2 := x.Shape
	if len(x.Shape) == 1 {
		x2 = []int{1, x.Shape[0]}
	}
	if len(x2) != 2 {
		return nil, fmt.Errorf("qlinear expects a 1-D or 2-D input, got rank %d", len(x.Shape))
	}
	m, k := x2[0], x2[1]
	if k != q.Cols {
		return nil, fmt.Errorf("shape mismatch in qlinear: input inner %d != weight inner %d", k, q.Cols)
	}
	n := q.Rows
	bpr := blocksPerRow(k, q.Block)
	rb := rowBytes(k)
	out := make([]float64, m*n)
	runChunks(m, workersFor(m*k*n), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			xRow := x.Data[i*k : i*k+k]
			cRow := i * n
			for j := 0; j < n; j++ {
				base := j * rb
				var y float64
				for b := 0; b < bpr; b++ {
					p0 := b * q.Block
					p1 := p0 + q.Block
					if p1 > k {
						p1 = k
					}
					var acc float64
					for p := p0; p < p1; p++ {
						acc += xRow[p] * float64(getNibble(q.Q, base, p))
					}
					y += acc * q.Scale[j*bpr+b]
				}
				out[cRow+j] = y
			}
		}
	})
	var outShape []int
	if len(x.Shape) == 1 {
		outShape = []int{n}
	} else {
		outShape = []int{m, n}
	}
	return &Tensor{Data: out, Shape: outShape}, nil
}
