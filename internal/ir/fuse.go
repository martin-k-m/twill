package ir

import (
	"fmt"
	"strings"
)

// Fusion: grouping the graph into the regions a backend launches one at a time.
//
// The measurement this pass exists for is not the 80 microseconds of GPU launch
// overhead in docs/CODEGEN.md. It is the CPU profile from the previous phase:
// about 18% of runtime spent allocating and zeroing intermediate buffers that
// are read once and discarded. A region is precisely a set of values that never
// get a buffer. Everything below follows from wanting that set to be as large
// as it can be while staying obviously correct.
//
// The rules are docs/CODEGEN.md section 3, restricted to what this first
// implementation actually does:
//
//   - An elementwise node absorbs any elementwise producer that has exactly one
//     use and is not a graph output. Recomputation is not attempted; a value
//     read twice is materialised, because that is the version that is obviously
//     correct and the cost comparison that would justify recomputing needs a
//     measurement nobody has taken yet.
//   - A reduction absorbs its elementwise producers the same way, so
//     mean(relu(ST - K)) computes an element and accumulates it without the
//     intermediate ever existing.
//   - A contraction is its own region. Epilogue fusion into a matmul is not
//     done here; an elementwise chain *after* a matmul is still one region, so
//     relu(x @ W + b) is two regions rather than the three-and-a-bit the
//     interpreter runs, but not the one docs/CODEGEN.md aims at.
//   - Structural ops (reshape, transpose, broadcast_to) are their own regions
//     and are materialised. Folding them into a consumer as an index remap is
//     the change that would collect the 14% docs/perf-baseline.md attributes to
//     TransposePerm, and it is not in this pass. Restricting absorption to
//     elementwise producers is what makes the index arithmetic in the emitter a
//     single effStrides read per input and therefore checkable by inspection.
type RegionKind uint8

const (
	// RegionBuffer is a leaf: a param or a const. It is a buffer that already
	// exists and no kernel computes it.
	RegionBuffer RegionKind = iota
	// RegionMap is an elementwise kernel: one loop over the output index.
	RegionMap
	// RegionReduce is a reduction with its elementwise producers folded into it.
	RegionReduce
	// RegionStructural is a copy under an index remap.
	RegionStructural
	// RegionContract is a matmul.
	RegionContract
)

func (k RegionKind) String() string {
	switch k {
	case RegionBuffer:
		return "buffer"
	case RegionMap:
		return "map"
	case RegionReduce:
		return "reduce"
	case RegionStructural:
		return "structural"
	case RegionContract:
		return "contract"
	}
	return "?"
}

// Region is one kernel launch.
type Region struct {
	Kind RegionKind
	Root Ref   // the node whose value the region produces
	Body []Ref // every node computed inside, in dependency order, ending at Root
	In   []Ref // values the region reads that it does not compute
	// Iter is the shape the region's loop walks. For a map that is the root's
	// own shape; for a reduce it is the shape being reduced over, which is the
	// root's operand shape.
	Iter []int
}

// Plan is a whole graph cut into regions.
type Plan struct {
	Graph   *Graph
	Regions []Region
	// Owner maps a node to the region that computes it. An absorbed node shares
	// its consumer's region; what separates the two is Materialised, which is
	// true only for a region's Root and for leaves.
	Owner []int
	// Materialised reports whether a node needs a buffer at run time. This is
	// the number the whole design is aimed at: the count of false entries is the
	// count of intermediates that no longer exist.
	Materialised []bool
}

// FuseOff is the plan with no fusion at all: every node is its own region. It
// exists because the differential harness in stage 3 runs the same program at
// every fusion setting and compares, and a setting where nothing fuses is the
// control.
func FuseOff(g *Graph) *Plan { return fuse(g, false) }

// Fuse is the greedy pass.
func Fuse(g *Graph) *Plan { return fuse(g, true) }

func fuse(g *Graph, greedy bool) *Plan {
	uses := g.Uses()
	p := &Plan{
		Graph:        g,
		Owner:        make([]int, len(g.Nodes)),
		Materialised: make([]bool, len(g.Nodes)),
	}
	for i := range p.Owner {
		p.Owner[i] = -1
	}
	absorbed := make([]bool, len(g.Nodes))
	isOut := make([]bool, len(g.Nodes))
	for _, r := range g.Out {
		isOut[r] = true
	}

	// A producer is absorbable when it is elementwise, has exactly one use, and
	// is not itself a graph output. Leaves are never absorbed: a param is a
	// buffer the caller supplied and a const is a buffer the graph carries.
	canAbsorb := func(r Ref) bool {
		if !greedy {
			return false
		}
		n := &g.Nodes[r]
		return n.Op.Class() == ClassElementwise && uses[r] == 1 && !isOut[r] && !absorbed[r]
	}

	// The walk is backwards. A region grows from its root towards its
	// producers, so a node has to be looked at only once its consumers are
	// already placed; going forwards would give every node its own region before
	// anything could absorb it. Regions come out roots-descending and are
	// reversed at the end, which puts them back in dependency order.
	for i := len(g.Nodes) - 1; i >= 0; i-- {
		n := &g.Nodes[i]
		if absorbed[i] {
			continue
		}
		if n.Op.Class() == ClassLeaf {
			p.Materialised[i] = true
			p.Regions = append(p.Regions, Region{
				Kind: RegionBuffer, Root: Ref(i), Body: []Ref{Ref(i)}, Iter: cloneShape(n.Shape),
			})
			continue
		}

		var kind RegionKind
		var iter []int
		switch n.Op.Class() {
		case ClassElementwise:
			kind, iter = RegionMap, cloneShape(n.Shape)
		case ClassReduction:
			kind, iter = RegionReduce, cloneShape(g.Nodes[n.In[0]].Shape)
		case ClassStructural:
			kind, iter = RegionStructural, cloneShape(n.Shape)
		case ClassContraction:
			kind, iter = RegionContract, cloneShape(n.Shape)
		}

		var body []Ref
		// Only map and reduce regions grow. A structural region is a copy and a
		// contraction is a tuned loop; absorbing into either is the epilogue and
		// index-remap work that is deliberately not in this pass.
		if kind == RegionMap || kind == RegionReduce {
			var grow func(r Ref)
			grow = func(r Ref) {
				for _, in := range g.Nodes[r].In {
					if canAbsorb(in) {
						absorbed[in] = true
						grow(in)
					}
				}
				body = append(body, r)
			}
			grow(Ref(i))
		} else {
			body = []Ref{Ref(i)}
		}

		inSet := map[Ref]bool{}
		for _, r := range body {
			inSet[r] = true
		}
		var ins []Ref
		seen := map[Ref]bool{}
		for _, r := range body {
			for _, x := range g.Nodes[r].In {
				if !inSet[x] && !seen[x] {
					seen[x] = true
					ins = append(ins, x)
				}
			}
		}

		p.Materialised[i] = true
		p.Regions = append(p.Regions, Region{
			Kind: kind, Root: Ref(i), Body: body, In: ins, Iter: iter,
		})
	}
	// Back into dependency order, then record who owns what.
	for l, r := 0, len(p.Regions)-1; l < r; l, r = l+1, r-1 {
		p.Regions[l], p.Regions[r] = p.Regions[r], p.Regions[l]
	}
	for ri := range p.Regions {
		for _, r := range p.Regions[ri].Body {
			p.Owner[r] = ri
		}
	}
	return p
}

// Stats summarises a plan. The interesting number is Eliminated: values the
// interpreter would have allocated a buffer for and the compiled version does
// not.
type Stats struct {
	Nodes        int
	Regions      int
	Kernels      int // regions that actually compute something
	Materialised int
	Eliminated   int
	Bytes        int // f64 bytes still allocated for intermediates
	BytesUnfused int
}

// Stats measures a plan.
func (p *Plan) Stats() Stats {
	s := Stats{Nodes: len(p.Graph.Nodes), Regions: len(p.Regions)}
	for i, n := range p.Graph.Nodes {
		if n.Op.Class() == ClassLeaf {
			continue
		}
		s.BytesUnfused += 8 * Numel(n.Shape)
		if p.Materialised[i] {
			s.Materialised++
			s.Bytes += 8 * Numel(n.Shape)
		} else {
			s.Eliminated++
		}
	}
	for _, r := range p.Regions {
		if r.Kind != RegionBuffer {
			s.Kernels++
		}
	}
	return s
}

func (s Stats) String() string {
	return fmt.Sprintf("%d nodes, %d kernels, %d/%d intermediates materialised, %d eliminated, %d B vs %d B unfused",
		s.Nodes, s.Kernels, s.Materialised, s.Materialised+s.Eliminated, s.Eliminated, s.Bytes, s.BytesUnfused)
}

// String renders the plan region by region.
func (p *Plan) String() string {
	var b strings.Builder
	for i, r := range p.Regions {
		if r.Kind == RegionBuffer {
			continue
		}
		fmt.Fprintf(&b, "region %d %s -> %%%d %s\n  body:", i, r.Kind, r.Root, ShapeString(r.Iter))
		for _, x := range r.Body {
			fmt.Fprintf(&b, " %%%d(%s)", x, p.Graph.Nodes[x].Op)
		}
		fmt.Fprintf(&b, "\n  in:")
		for _, x := range r.In {
			fmt.Fprintf(&b, " %%%d", x)
		}
		b.WriteString("\n")
	}
	b.WriteString(p.Stats().String() + "\n")
	return b.String()
}
