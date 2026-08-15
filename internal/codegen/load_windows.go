//go:build windows

package codegen

import (
	"fmt"
	"syscall"
	"unsafe"
)

// entry is a loaded twill_run.
//
// The call goes through syscall rather than cgo deliberately. cgo would make
// the Go toolchain need a C compiler to *build twill*, which is exactly the
// dependency docs/DECISIONS.md entry 7 says the project does not want; going
// through syscall keeps the C compiler a run-time option that a machine either
// has or does not.
type entry struct {
	dll  *syscall.DLL
	proc *syscall.Proc
}

func loadEntry(path string) (*entry, error) {
	dll, err := syscall.LoadDLL(path)
	if err != nil {
		return nil, fmt.Errorf("codegen: loading %s: %w", path, err)
	}
	proc, err := dll.FindProc("twill_run")
	if err != nil {
		dll.Release()
		return nil, fmt.Errorf("codegen: %s has no twill_run: %w", path, err)
	}
	return &entry{dll: dll, proc: proc}, nil
}

// call runs the kernel over the arena. arena is a Go slice and its backing
// array is passed by address; that is safe here because Go's heap does not move
// objects and the syscall package keeps the argument alive for the duration of
// the call. The pointer is one level deep, so there is no Go-pointer-to-
// Go-pointer for the runtime to object to.
func (e *entry) call(arena []float64) error {
	if len(arena) == 0 {
		return nil
	}
	_, _, err := e.proc.Call(uintptr(unsafe.Pointer(&arena[0])))
	// Proc.Call always reports the thread's last error, which is stale rather
	// than meaningful for a function that returns void and sets nothing.
	_ = err
	return nil
}

func (e *entry) close() { e.dll.Release() }
