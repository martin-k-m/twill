package tensor

import "testing"

func makeTensor(shape []int) *Tensor {
	n := numel(shape)
	data := make([]float64, n)
	for i := range data {
		data[i] = float64(i%7) * 0.5
	}
	return New(data, shape)
}

func BenchmarkMatMul64(b *testing.B) {
	a := makeTensor([]int{64, 64})
	c := makeTensor([]int{64, 64})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatMul(a, c); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBroadcastAdd(b *testing.B) {
	m := makeTensor([]int{128, 128})
	row := makeTensor([]int{128})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Add(m, row); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSoftmax(b *testing.B) {
	x := makeTensor([]int{256, 32})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Softmax(x, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTrainStep approximates one gradient step of a small linear model.
func BenchmarkTrainStep(b *testing.B) {
	x := makeTensor([]int{32})
	target := Scalar(1.0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := Leaf(makeTensor([]int{32}).Data, []int{32})
		pred := Sum(mustMul(w, x))
		diff, _ := Sub(pred, target)
		loss := Square(diff)
		if err := loss.Backward(); err != nil {
			b.Fatal(err)
		}
	}
}

func mustMul(a, b *Tensor) *Tensor {
	r, err := Mul(a, b)
	if err != nil {
		panic(err)
	}
	return r
}
