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

// sumChunk is the fixed block size for parallelSum. Blocking by a fixed size
// (not by core count) makes the result deterministic regardless of how many
// goroutines run, and is slightly more accurate than a naive running sum.
const sumChunk = 4096

// parallelSum adds a slice deterministically: fixed-size blocks are summed
// independently, then their partials are combined in order. The result is the
// same on any number of cores.
func parallelSum(data []float64) float64 {
	n := len(data)
	if n < minParallel {
		s := 0.0
		for _, x := range data {
			s += x
		}
		return s
	}
	nblocks := (n + sumChunk - 1) / sumChunk
	partials := make([]float64, nblocks)
	runChunks(nblocks, workersFor(n), func(lo, hi int) {
		for blk := lo; blk < hi; blk++ {
			start := blk * sumChunk
			end := start + sumChunk
			if end > n {
				end = n
			}
			s := 0.0
			for i := start; i < end; i++ {
				s += data[i]
			}
			partials[blk] = s
		}
	})
	total := 0.0
	for _, p := range partials {
		total += p
	}
	return total
}
