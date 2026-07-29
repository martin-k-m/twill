package tensor

import "testing"

// Large-tensor benchmarks — run with `-cpu=1,8` to see the multicore scaling
// that matters for Monte-Carlo and backtesting workloads.

func BenchmarkExpLarge(b *testing.B) {
	x := makeTensor([]int{500000})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Exp(x)
	}
}

func BenchmarkAddLarge(b *testing.B) {
	x := makeTensor([]int{500000})
	y := makeTensor([]int{500000})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Add(x, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMul256(b *testing.B) {
	x := makeTensor([]int{256, 256})
	y := makeTensor([]int{256, 256})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatMul(x, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSumLarge(b *testing.B) {
	x := makeTensor([]int{1000000})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Sum(x)
	}
}
