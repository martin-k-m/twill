package tensor

import (
	"runtime"
	"sync"
)

// minParallel is the work size below which an op runs serially — goroutine
// overhead isn't worth it for small tensors (typical training parameters).
const minParallel = 8192

// runChunks splits [0, n) into contiguous chunks across `workers` goroutines and
// waits for them. body(lo, hi) must write only outputs in [lo, hi), so the
// result is race-free and bit-identical to a serial run.
func runChunks(n, workers int, body func(lo, hi int)) {
	if workers < 2 || n < 2 {
		body(0, n)
		return
	}
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// parallelFor runs body over [0, n) across cores when n is large enough that
// the parallelism pays off. Parallelism never changes a program's output.
func parallelFor(n int, body func(lo, hi int)) {
	workers := 1
	if n >= minParallel {
		workers = runtime.GOMAXPROCS(0)
	}
	runChunks(n, workers, body)
}

// workers returns the goroutine count to use for a task of the given total work.
func workersFor(work int) int {
	if work < minParallel {
		return 1
	}
	return runtime.GOMAXPROCS(0)
}
