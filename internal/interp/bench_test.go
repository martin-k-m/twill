package interp_test

import (
	"testing"

	"github.com/martin-k-m/raster/internal/interp"
)

// A representative training loop: linear regression by gradient descent.
const trainingProgram = `
let X = [[1.0, 1.0], [2.0, 1.0], [1.0, 3.0], [3.0, 2.0], [0.0, 4.0]]
let y = [-0.5, 1.5, -6.5, 0.5, -11.5]
fn loss(w, b) {
  let err = X @ w + b - y
  mean(err * err)
}
let w = [0.0, 0.0]
let b = 0.0
for step in range(100) {
  let g = grads(loss)(w, b)
  w = w - g[0] * 0.05
  b = b - g[1] * 0.05
}
`

// A loop that does nothing but scalar arithmetic, which is where the
// interpreter's allocation cost shows up undiluted by any real work.
const scalarLoopProgram = `
let acc = 0.0
for i in range(3000000) {
  acc = acc + 1.0
}
`

func BenchmarkScalarLoop(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ip := interp.New(func(string) {})
		if _, err := ip.Run(scalarLoopProgram); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTrainingLoop(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ip := interp.New(func(string) {})
		if _, err := ip.Run(trainingProgram); err != nil {
			b.Fatal(err)
		}
	}
}
