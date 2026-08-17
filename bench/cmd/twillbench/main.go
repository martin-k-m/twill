// Command twillbench times twill workloads and reports a distribution rather
// than an average.
//
// Each workload is a .tw file whose top level does the setup and whose final
// expression is a nullary closure holding the work. The harness runs the top
// level once, then calls that closure repeatedly, recording the wall time of
// every call. Setup therefore costs nothing in the reported numbers, and what
// is timed is the evaluation of the work itself: the same thing torch_bench.py
// times on the PyTorch side, so the two are comparable.
//
// It reports the median and the p99, not the mean. A mean over an interpreted
// workload on a laptop is dominated by whatever the operating system decided to
// do during the run; the median says what a typical call costs and the p99 says
// how bad the tail is, and neither can be moved by one outlier.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/twill-lang/twill/internal/checker"
	"github.com/twill-lang/twill/internal/interp"
	"github.com/twill-lang/twill/internal/parser"
	"github.com/twill-lang/twill/internal/value"
)

type record struct {
	Workload string  `json:"workload"`
	Impl     string  `json:"impl"`
	Runs     int     `json:"runs"`
	Warmup   int     `json:"warmup"`
	Inner    int     `json:"inner"`
	Procs    int     `json:"gomaxprocs"`
	Best     bool    `json:"best_of_sweep,omitempty"`
	MedianMS float64 `json:"median_ms"`
	P99MS    float64 `json:"p99_ms"`
	MinMS    float64 `json:"min_ms"`
	MaxMS    float64 `json:"max_ms"`
	Result   string  `json:"result"`
}

func main() {
	var (
		dir    = flag.String("dir", "bench/workloads", "directory of .tw workloads")
		runs   = flag.Int("runs", 30, "timed runs per workload")
		warmup = flag.Int("warmup", 5, "untimed warmup runs per workload")
		only   = flag.String("only", "", "run only workloads whose name contains this")
		out    = flag.String("out", "", "write JSON results here as well as stdout")
		procs  = flag.String("procs", "1,2,4,8,16", "comma-separated GOMAXPROCS values to sweep; each workload is reported at its best")
	)
	flag.Parse()

	files, err := filepath.Glob(filepath.Join(*dir, "*.tw"))
	if err != nil || len(files) == 0 {
		fmt.Fprintf(os.Stderr, "twillbench: no workloads under %s\n", *dir)
		os.Exit(1)
	}
	sort.Strings(files)

	var procList []int
	for _, p := range strings.Split(*procs, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil || n < 1 {
			fmt.Fprintf(os.Stderr, "twillbench: bad -procs value %q\n", p)
			os.Exit(2)
		}
		procList = append(procList, n)
	}
	if len(procList) == 0 {
		procList = []int{runtime.NumCPU()}
	}

	fmt.Printf("# twillbench: procs_swept=%v numcpu=%d go=%s %s/%s\n",
		procList, runtime.NumCPU(), runtime.Version(), runtime.GOOS, runtime.GOARCH)
	fmt.Printf("%-26s %5s %6s %11s %11s %11s %11s  %s\n",
		"workload", "proc", "inner", "median_ms", "p99_ms", "min_ms", "max_ms", "result")

	var records []record
	for _, f := range files {
		name := strings.TrimSuffix(filepath.Base(f), ".tw")
		if *only != "" && !strings.Contains(name, *only) {
			continue
		}
		// The whole sweep is run and printed, and the best row is marked. A
		// thread count is not a neutral choice: PyTorch's pool collapses on
		// small kernels here, and reporting either side at a count that happens
		// to suit it is how a benchmark ends up saying what its author wanted.
		var sweep []record
		for _, np := range procList {
			runtime.GOMAXPROCS(np)
			r, err := benchOne(name, f, *runs, *warmup)
			if err != nil {
				fmt.Fprintf(os.Stderr, "twillbench: %s: %v\n", name, err)
				os.Exit(1)
			}
			r.Procs = np
			sweep = append(sweep, r)
		}
		best := 0
		for i, r := range sweep {
			if r.MedianMS < sweep[best].MedianMS {
				best = i
			}
		}
		sweep[best].Best = true
		for i, r := range sweep {
			mark := ""
			if i == best {
				mark = " <- best"
			}
			fmt.Printf("%-26s %5d %6d %11.4f %11.4f %11.4f %11.4f  %s%s\n",
				r.Workload, r.Procs, r.Inner, r.MedianMS, r.P99MS, r.MinMS, r.MaxMS, r.Result, mark)
		}
		records = append(records, sweep...)
	}

	if *out != "" {
		b, _ := json.MarshalIndent(records, "", "  ")
		if err := os.WriteFile(*out, b, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "twillbench: writing %s: %v\n", *out, err)
			os.Exit(1)
		}
	}
}

func benchOne(name, path string, runs, warmup int) (record, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return record{}, err
	}
	// Shape-check the workload, so a benchmark cannot quietly be measuring a
	// program that twill itself would refuse to run.
	prog, perr := parser.Parse(string(src))
	if perr != nil {
		return record{}, perr
	}
	if diags := checker.Check(prog); len(diags) > 0 {
		return record{}, fmt.Errorf("line %d: %s", diags[0].Line, diags[0].Msg)
	}

	// Discard the workload's own output: a workload prints nothing, but a print
	// left in during development should not land in the results table.
	ip := interp.New(func(string) {})
	work, err := ip.Run(string(src))
	if err != nil {
		return record{}, err
	}
	if _, ok := work.(*value.Closure); !ok {
		return record{}, fmt.Errorf("the last expression must be a nullary closure holding the work, got %T", work)
	}

	call := func() (v value.Value, err error) {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("%v", r)
			}
		}()
		return ip.Apply(work, nil, 0), nil
	}

	var last value.Value
	for i := 0; i < warmup; i++ {
		if last, err = call(); err != nil {
			return record{}, err
		}
	}

	// Calibrate an inner repeat count. Windows' wall clock has roughly
	// half-millisecond granularity, so timing a single call that takes 40us
	// reports either 0 or 1ms and nothing in between. Repeating the call until
	// the timed span is comfortably above the clock's resolution and dividing
	// gives a real number instead of a quantised one.
	//
	// The cost is that a workload with inner > 1 reports the distribution of
	// batch means rather than of individual calls, which understates the tail.
	// The inner count is reported alongside every row so a reader can see which
	// rows that applies to.
	const targetSpan = 20 * time.Millisecond
	inner := 1
	for {
		start := time.Now()
		for i := 0; i < inner; i++ {
			if last, err = call(); err != nil {
				return record{}, err
			}
		}
		span := time.Since(start)
		if span >= targetSpan || inner >= 1<<16 {
			break
		}
		if span <= 0 {
			inner *= 8
			continue
		}
		next := int(float64(inner) * float64(targetSpan) / float64(span) * 1.2)
		if next <= inner {
			next = inner * 2
		}
		inner = next
	}

	samples := make([]float64, 0, runs)
	for i := 0; i < runs; i++ {
		start := time.Now()
		for j := 0; j < inner; j++ {
			if last, err = call(); err != nil {
				return record{}, err
			}
		}
		elapsed := float64(time.Since(start).Nanoseconds()) / 1e6
		samples = append(samples, elapsed/float64(inner))
	}
	sort.Float64s(samples)

	return record{
		Workload: name, Impl: "twill", Runs: runs, Warmup: warmup, Inner: inner,
		MedianMS: quantile(samples, 0.5),
		P99MS:    quantile(samples, 0.99),
		MinMS:    samples[0],
		MaxMS:    samples[len(samples)-1],
		// The result is printed so the PyTorch side can be checked to be doing
		// the same arithmetic, not merely arithmetic of the same shape.
		Result: summarise(last),
	}, nil
}

// quantile is the nearest-rank quantile of an already-sorted sample. Nearest
// rank rather than interpolation: with 30 samples an interpolated p99 is mostly
// invention, and the honest p99 of 30 runs is the slowest one.
func quantile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(q * float64(len(sorted)))
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

// summarise renders a workload's return value compactly: a scalar in full, a
// tensor as its shape and the sum of its elements, which is enough to tell two
// implementations apart without printing a million numbers.
func summarise(v value.Value) string {
	s := value.Format(v)
	if len(s) > 90 {
		return s[:87] + "..."
	}
	return s
}
