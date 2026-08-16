package codegen

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// ErrNoCompiler is returned when no C compiler can be found. It is a distinct
// error because every caller treats it the same way and differently from a real
// failure: the compiled path is unavailable, the interpreter runs, nothing is
// wrong. twill's single dependency-free binary is a stated property
// (docs/DECISIONS.md entry 7), so a missing compiler is an expected state of
// the world and not a defect.
var ErrNoCompiler = errors.New("codegen: no C compiler found (set TWILL_CC to one)")

// ccFlags are the flags the emitted code's correctness depends on.
//
// -ffp-contract=off is the one that matters most. With contraction on, the C
// compiler is free to fuse a multiply and an add into an FMA, which rounds once
// where Go rounds twice. Every reduction and every a*b+c in the emitted code
// would then differ from the interpreter in the last bits, and the bit-exact
// comparison in the differential tests would fail for a reason that has nothing
// to do with the compiler being wrong.
//
// -fno-fast-math is stated explicitly rather than relied on as a default,
// because a machine with it in CFLAGS would silently reassociate the reductions
// docs/DECISIONS.md entry 6 requires not be reassociated.
var ccFlags = []string{
	"-O2", "-std=c99", "-shared", "-fPIC",
	"-ffp-contract=off", "-fno-fast-math", "-fno-unsafe-math-optimizations",
}

// FindCompiler returns the C compiler to use, honouring TWILL_CC.
//
// The answer is found once per process. On Windows a LookPath miss walks every
// PATH entry against every executable extension before it fails, and profiling
// a traced run put this at 79% of it: three compilations, three searches, and
// the search cost more than the compiler did.
func FindCompiler() (string, error) {
	ccOnce.Do(func() { ccPath, ccErr = findCompiler() })
	return ccPath, ccErr
}

var (
	ccOnce sync.Once
	ccPath string
	ccErr  error
)

func findCompiler() (string, error) {
	if cc := os.Getenv("TWILL_CC"); cc != "" {
		if p, err := exec.LookPath(cc); err == nil {
			return p, nil
		}
		return "", fmt.Errorf("codegen: TWILL_CC=%q is not executable", cc)
	}
	for _, name := range []string{"gcc", "clang", "cc"} {
		if p, err := exec.LookPath(name); err == nil {
			return p, nil
		}
	}
	return "", ErrNoCompiler
}

func libExt() string {
	if runtime.GOOS == "windows" {
		return ".dll"
	}
	return ".so"
}

// buildShared compiles src and returns the path of the shared library. Output
// is cached under the user cache directory keyed by a hash of the source and
// the flags, so a training loop that recompiles the same trace on every
// iteration pays the compiler once per process lifetime and once per machine.
func buildShared(src string) (string, error) {
	cc, err := FindCompiler()
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256([]byte(src + "\x00" + strings.Join(ccFlags, " ") + "\x00" + cc))
	key := hex.EncodeToString(sum[:12])

	base, err := os.UserCacheDir()
	if err != nil || base == "" {
		base = os.TempDir()
	}
	dir := filepath.Join(base, "twill-codegen")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	lib := filepath.Join(dir, "k"+key+libExt())
	if _, err := os.Stat(lib); err == nil {
		return lib, nil
	}

	cfile := filepath.Join(dir, "k"+key+".c")
	if err := os.WriteFile(cfile, []byte(src), 0o644); err != nil {
		return "", err
	}
	// Compile to a temporary name and rename, so two processes racing on the
	// same trace cannot leave a half-written library for the other to load.
	tmp, err := os.CreateTemp(dir, "build*"+libExt())
	if err != nil {
		return "", err
	}
	tmpName := tmp.Name()
	tmp.Close()
	args := append(append([]string{}, ccFlags...), "-o", tmpName, cfile, "-lm")
	cmd := exec.Command(cc, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		os.Remove(tmpName)
		return "", fmt.Errorf("codegen: %s failed: %v\n%s", filepath.Base(cc), err, out)
	}
	if err := os.Rename(tmpName, lib); err != nil {
		// A rename over an existing file fails on Windows if another process
		// has it mapped; if the target is there, someone else won the race.
		if _, statErr := os.Stat(lib); statErr == nil {
			os.Remove(tmpName)
			return lib, nil
		}
		return "", err
	}
	return lib, nil
}
