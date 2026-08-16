package trace

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"strconv"
	"sync"

	"github.com/martin-k-m/twill/internal/codegen"
	"github.com/martin-k-m/twill/internal/ir"
)

// Cache maps a graph's structure to the program compiled from it.
//
// docs/CODEGEN.md section 2 says this is what makes the design pay, and the
// reason is arithmetic rather than taste. Compiling a graph means emitting C,
// starting gcc, waiting for it, and loading a DLL, which is tens to hundreds of
// milliseconds; running the result is microseconds. A tracer without a cache
// loses on every program. What makes a cache possible at all is that a loop body
// traces to the same graph on every iteration, so a training loop compiles once
// and calls in thousands of times.
//
// The key is the graph's structure: opcodes, operand indices, shapes, attributes
// and constant data. Parameter *values* are deliberately not in it, which is why
// only Params and no Consts are traced from a program: a learning rate that
// changes between iterations must not compile a second kernel.
type Cache struct {
	mu   sync.Mutex
	m    map[string]*entry
	fuse bool
}

type entry struct {
	prog *codegen.Program
	bad  bool // compilation was tried and failed; do not try again
}

// NewCache returns an empty cache. TWILL_TRACE_FUSE=0 turns the fusion pass off,
// which is the control setting the differential harness compares against: fused
// and unfused must agree, and stage 3 already established they do at the IR
// level.
func NewCache() *Cache {
	fuse := true
	if v := os.Getenv("TWILL_TRACE_FUSE"); v != "" {
		b, err := strconv.ParseBool(v)
		fuse = err == nil && b
	}
	return &Cache{m: map[string]*entry{}, fuse: fuse}
}

// Get returns the compiled program for a graph, compiling it on the first sight
// and remembering a failure so that a graph the backend refuses is not offered
// to it again on every iteration of a loop.
func (c *Cache) Get(g *ir.Graph, st *Stats) (*codegen.Program, bool) {
	k := Key(g)
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.m[k]; ok {
		if e.bad {
			return nil, false
		}
		st.CacheHit++
		return e.prog, true
	}
	st.CacheMiss++
	p, err := codegen.Compile(g, codegen.Options{Fuse: c.fuse})
	if err != nil {
		c.m[k] = &entry{bad: true}
		return nil, false
	}
	c.m[k] = &entry{prog: p}
	return p, true
}



// Key hashes everything about a graph that changes what the emitted code does,
// and nothing that does not. Parameter values are not in it; constant values
// are, because a constant is baked into the C.
func Key(g *ir.Graph) string {
	h := sha256.New()
	var buf [8]byte
	put := func(x int) {
		binary.LittleEndian.PutUint64(buf[:], uint64(x))
		h.Write(buf[:])
	}
	putf := func(x float64) {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(x))
		h.Write(buf[:])
	}
	putShape := func(s []int) {
		put(len(s))
		for _, d := range s {
			put(d)
		}
	}
	put(len(g.Params))
	for _, p := range g.Params {
		putShape(p.Shape)
	}
	put(len(g.Consts))
	for _, c := range g.Consts {
		putShape(c.Shape)
		for _, x := range c.Data {
			putf(x)
		}
	}
	put(len(g.Nodes))
	for _, n := range g.Nodes {
		put(int(n.Op))
		put(len(n.In))
		for _, r := range n.In {
			put(int(r))
		}
		putShape(n.Shape)
		put(n.Attr.Index)
		put(n.Attr.Axis)
		putShape(n.Attr.Shape)
		putf(n.Attr.F)
		putf(n.Attr.G)
	}
	put(len(g.Out))
	for _, r := range g.Out {
		put(int(r))
	}
	return hex.EncodeToString(h.Sum(nil))
}
