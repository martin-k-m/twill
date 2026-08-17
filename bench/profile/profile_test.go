// Package profile exists to give the profiler something to attach to.
//
// `go test -cpuprofile` needs a test binary, so the profiling target is a test
// rather than a command. It runs the Monte Carlo pricer, the program the README
// leads with, forward and then differentiated, long enough for the sampling
// profiler to have something to say.
//
// Regenerate the profile with:
//
//	go test ./bench/profile/ -run TestProfileMonteCarlo -count=1 \
//	    -cpuprofile mc.prof -o interp.test.exe
//	go tool pprof -top -nodecount=20 interp.test.exe mc.prof
package profile

import (
	"os"
	"testing"

	"github.com/twill-lang/twill/internal/interp"
	"github.com/twill-lang/twill/internal/value"
)

// iterations is small enough that the test is not a burden in a normal `go
// test ./...` and large enough that a 100Hz sampling profiler collects a few
// hundred samples when one is asked for.
const iterations = 40

func runWorkload(t *testing.T, path string) {
	t.Helper()
	src, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("workload %s not found: %v", path, err)
	}
	ip := interp.New(func(string) {})
	work, err := ip.Run(string(src))
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	if _, ok := work.(*value.Closure); !ok {
		t.Fatalf("%s: last expression is not a nullary closure", path)
	}
	for i := 0; i < iterations; i++ {
		ip.Apply(work, nil, 0)
	}
}

func TestProfileMonteCarlo(t *testing.T) {
	runWorkload(t, "../workloads/mc_option_grad.tw")
}

func TestProfileMLPTrainStep(t *testing.T) {
	runWorkload(t, "../workloads/mlp_train_step.tw")
}
