//go:build !windows

package codegen

// Loading a shared library at run time without cgo has no portable answer
// outside Windows, where the syscall package already exposes LoadDLL. On the
// other platforms the emitter still runs and its output can still be compiled
// and inspected; what is missing is the dial-in, so Compile reports this and
// the caller uses the interpreter.
//
// Two ways out exist and neither is in scope for this stage: build the backend
// under a cgo build tag, or emit and dlopen through a small assembly trampoline.
// The first is the obvious one and costs the dependency-free build only on the
// tag.

type entry struct{}

func loadEntry(path string) (*entry, error) { return nil, ErrNoLoader }

func (e *entry) call(arena []float64) error { return ErrNoLoader }

func (e *entry) close() {}
