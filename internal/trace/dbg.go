package trace

import "fmt"

// verify is a debug assertion used while bringing the tracer up.
func (t *Tracer) verify(where string) {
	for i, ph := range t.order {
		if ph.Data == nil {
			panic(fmt.Sprintf("trace: %s left placeholder %d (node %d) unpatched", where, i, t.phRef[i]))
		}
	}
}
