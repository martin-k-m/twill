package tensor

import (
	"math"
	"runtime"
	"testing"
)

// The README claims parallelism never changes a result. The implementation is
// built for it, fixed 4096-element blocks combined in block order, but until
// this test nothing ran the same reduction under different core counts and
// compared. A claim about concurrency that no test varies concurrency for is a
// claim about the code someone read, not the code that runs.
func TestReductionsAreIdenticalAtEveryCoreCount(t *testing.T) {
	sizes := []int{minParallel - 1, minParallel, sumChunk*3 + 7, 1 << 20}
	procs := []int{1, 2, 3, 16}

	original := runtime.GOMAXPROCS(0)
	t.Cleanup(func() { runtime.GOMAXPROCS(original) })

	for _, n := range sizes {
		data := make([]float64, n)
		for i := range data {
			// Mixed magnitudes, so a changed summation order shows up as a
			// changed answer rather than being absorbed by the rounding.
			data[i] = math.Sin(float64(i)) * math.Pow(10, float64(i%17)-8)
		}

		var want float64
		for i, p := range procs {
			runtime.GOMAXPROCS(p)
			got := parallelSum(data)
			if i == 0 {
				want = got
				continue
			}
			if got != want {
				t.Errorf("n=%d: sum at GOMAXPROCS=%d is %v, at %d it is %v, difference %g",
					n, p, got, procs[0], want, got-want)
			}
		}
	}
}

// Sum is the exported path onto parallelSum, so the property has to hold there
// too or the claim is about an internal function nobody calls.
func TestSumIsIdenticalAtEveryCoreCount(t *testing.T) {
	original := runtime.GOMAXPROCS(0)
	t.Cleanup(func() { runtime.GOMAXPROCS(original) })

	n := 1 << 20
	data := make([]float64, n)
	for i := range data {
		data[i] = math.Cos(float64(i)) * float64(i%1000)
	}
	a := New(data, []int{n})

	runtime.GOMAXPROCS(1)
	serial := Sum(a)
	for _, p := range []int{2, 4, 16} {
		runtime.GOMAXPROCS(p)
		if got := Sum(a); got.Data[0] != serial.Data[0] {
			t.Errorf("Sum at GOMAXPROCS=%d is %v, serial is %v", p, got.Data[0], serial.Data[0])
		}
	}
}
