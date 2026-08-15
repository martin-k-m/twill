// tracestats runs each .tw file given and reports what the tracer did with it,
// so "how much of the language compiles" is a measurement rather than a guess.
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/martin-k-m/twill/internal/interp"
)

func main() {
	files := os.Args[1:]
	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "usage: tracestats <file.tw>...")
		os.Exit(2)
	}

	fmt.Printf("%-34s %7s %7s %8s %8s %7s %7s %8s %8s\n",
		"program", "nodes", "scopes", "compiled", "replayed", "hit", "miss", "gradfast", "gradslow")

	var traced, total int
	for _, f := range files {
		ip := interp.New(func(string) {})
		ip.SetTracing(true)
		if _, _, err := ip.RunFileMain(f, nil); err != nil {
			fmt.Printf("%-34s failed: %v\n", filepath.Base(f), err)
			continue
		}
		s := ip.TraceStats()
		total++
		if s.Nodes > 0 {
			traced++
		}
		fmt.Printf("%-34s %7d %7d %8d %8d %7d %7d %8d %8d\n",
			filepath.Base(f), s.Nodes, s.Scopes, s.Compiled, s.Replayed,
			s.CacheHit, s.CacheMiss, s.GradFast, s.GradSlow)
	}
	fmt.Printf("\n%d of %d programs produced traced nodes\n", traced, total)
}
