// Command checkbench times the front end: lexing, parsing and static shape
// checking, separately, over a corpus of real .tw files.
//
// This is the one measurement with no PyTorch counterpart, because PyTorch has
// nothing to compare it against. Shapes in PyTorch are discovered by running
// the program, so the equivalent cost is not a compile step that can be timed,
// it is the wait until the offending line executes. What can be compared is
// what the two buy you, and that is a question about the run, not about the
// clock.
//
// Parse and check are timed apart because they answer different questions. The
// parse time is what any tool touching the source pays. The check time is what
// the guarantee costs on top, and it is the number that decides whether the
// checker can run on every keystroke in an editor or only on save.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/twill-lang/twill/internal/checker"
	"github.com/twill-lang/twill/internal/lexer"
	"github.com/twill-lang/twill/internal/parser"
)

type row struct {
	File        string  `json:"file"`
	Bytes       int     `json:"bytes"`
	Lines       int     `json:"lines"`
	LexMedianMS float64 `json:"lex_median_ms"`
	ParseMedMS  float64 `json:"parse_median_ms"`
	ParseP99MS  float64 `json:"parse_p99_ms"`
	CheckMedMS  float64 `json:"check_median_ms"`
	CheckP99MS  float64 `json:"check_p99_ms"`
	Diags       int     `json:"diagnostics"`
}

func main() {
	var (
		globs  = flag.String("globs", "examples/*.tw,std/*.tw,src/*.tw", "comma-separated globs of files to check")
		runs   = flag.Int("runs", 30, "timed runs per file per phase")
		warmup = flag.Int("warmup", 5, "untimed warmup runs")
		out    = flag.String("out", "", "write JSON results here as well as stdout")
		top    = flag.Int("top", 15, "print this many of the largest files individually")
	)
	flag.Parse()

	var files []string
	for _, g := range strings.Split(*globs, ",") {
		m, err := filepath.Glob(strings.TrimSpace(g))
		if err != nil {
			fmt.Fprintf(os.Stderr, "checkbench: bad glob %q: %v\n", g, err)
			os.Exit(2)
		}
		files = append(files, m...)
	}
	sort.Strings(files)
	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "checkbench: no files matched")
		os.Exit(1)
	}

	var rows []row
	var totalBytes, totalLines int
	var sumLex, sumParse, sumCheck float64
	for _, f := range files {
		r, err := timeOne(f, *runs, *warmup)
		if err != nil {
			// A file that does not parse is not a benchmark failure; src/ holds
			// systems-mode sources the numeric front end reads fine, but a stray
			// unparseable file should be reported and skipped, not fatal.
			fmt.Fprintf(os.Stderr, "checkbench: skipping %s: %v\n", f, err)
			continue
		}
		rows = append(rows, r)
		totalBytes += r.Bytes
		totalLines += r.Lines
		sumLex += r.LexMedianMS
		sumParse += r.ParseMedMS
		sumCheck += r.CheckMedMS
	}

	fmt.Printf("# checkbench: %d files, %d lines, %d bytes\n", len(rows), totalLines, totalBytes)
	fmt.Printf("# whole corpus, summed medians: lex %.2f ms, parse %.2f ms, check %.2f ms\n",
		sumLex, sumParse, sumCheck)
	if totalLines > 0 {
		fmt.Printf("# per 1000 lines: lex %.3f ms, parse %.3f ms, check %.3f ms\n",
			sumLex*1000/float64(totalLines), sumParse*1000/float64(totalLines),
			sumCheck*1000/float64(totalLines))
	}

	byLines := append([]row(nil), rows...)
	sort.Slice(byLines, func(i, j int) bool { return byLines[i].Lines > byLines[j].Lines })
	fmt.Printf("\n%-34s %6s %9s %9s %9s %9s %6s\n",
		"file", "lines", "lex_ms", "parse_ms", "check_ms", "check_p99", "diags")
	for i, r := range byLines {
		if i >= *top {
			break
		}
		fmt.Printf("%-34s %6d %9.4f %9.4f %9.4f %9.4f %6d\n",
			r.File, r.Lines, r.LexMedianMS, r.ParseMedMS, r.CheckMedMS, r.CheckP99MS, r.Diags)
	}

	if *out != "" {
		b, _ := json.MarshalIndent(rows, "", "  ")
		if err := os.WriteFile(*out, b, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "checkbench: writing %s: %v\n", *out, err)
			os.Exit(1)
		}
	}
}

func timeOne(path string, runs, warmup int) (row, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return row{}, err
	}
	text := string(src)
	prog, perr := parser.Parse(text)
	if perr != nil {
		return row{}, perr
	}

	lexTimes := timePhase(runs, warmup, func() { lexer.Tokenize(text) })
	parseTimes := timePhase(runs, warmup, func() { parser.Parse(text) })
	// The checker is timed on an already-parsed program, so the number is the
	// cost of the analysis and not of the parse it would otherwise include.
	checkTimes := timePhase(runs, warmup, func() { checker.Check(prog) })

	return row{
		File:        filepath.ToSlash(path),
		Bytes:       len(src),
		Lines:       strings.Count(text, "\n") + 1,
		LexMedianMS: median(lexTimes),
		ParseMedMS:  median(parseTimes),
		ParseP99MS:  quantile(parseTimes, 0.99),
		CheckMedMS:  median(checkTimes),
		CheckP99MS:  quantile(checkTimes, 0.99),
		Diags:       len(checker.Check(prog)),
	}, nil
}

func timePhase(runs, warmup int, f func()) []float64 {
	for i := 0; i < warmup; i++ {
		f()
	}
	// Front-end phases on a small file run in microseconds, well under the
	// Windows clock's resolution, so the same inner-repeat calibration the
	// workload harness uses applies here.
	const targetSpan = 20 * time.Millisecond
	inner := 1
	for {
		start := time.Now()
		for i := 0; i < inner; i++ {
			f()
		}
		span := time.Since(start)
		if span >= targetSpan || inner >= 1<<20 {
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
	out := make([]float64, 0, runs)
	for i := 0; i < runs; i++ {
		start := time.Now()
		for j := 0; j < inner; j++ {
			f()
		}
		out = append(out, float64(time.Since(start).Nanoseconds())/1e6/float64(inner))
	}
	sort.Float64s(out)
	return out
}

func median(sorted []float64) float64 { return quantile(sorted, 0.5) }

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
