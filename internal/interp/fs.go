package interp

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/twill-lang/twill/internal/value"
)

// The filesystem surface beyond reading and writing a whole file: making and
// removing things, asking what a path is without reading it, and a temporary
// directory to put a test's output in.
//
// Every one of these can fail for a reason the caller may reasonably handle --
// a missing directory, a permission, a file that is already gone -- so each
// returns a Res and none of them aborts. The predicates (`path_exists`,
// `path_is_dir`) return a Bool instead, because "it is not there" is the answer
// rather than a failure.
//
// docs/needs.md NEEDS-91 and NEEDS-92. Before these, `std/io.exists` read the
// whole file to find out whether it was there, a test could not clean up after
// itself, and selvedge has seven leftover `tmp_*` files committed to its
// repository because of it.

func fsOk(v value.Value) value.Value {
	return &value.Variant{Name: "Ok", Payload: v, HasPayload: true}
}

func fsErr(err error) value.Value {
	return &value.Variant{Name: "Err", Payload: value.Str(err.Error()), HasPayload: true}
}

// registerFS defines the filesystem and path builtins. def and asStr are the
// registrar and the string reader from builtins.go.
func (ip *Interp) registerFS(def func(string, int, bool, func([]value.Value) (value.Value, error)), asStr func(value.Value) (string, bool)) {
	// pathArg reads argument i as a path resolved against the running file, the
	// way read_file and save already resolve theirs.
	pathArg := func(a []value.Value, i int, who string) (string, error) {
		s, ok := asStr(a[i])
		if !ok {
			return "", fmt.Errorf("%s expects a path", who)
		}
		return ip.resolvePath(s), nil
	}

	def("path_exists", 1, false, func(a []value.Value) (value.Value, error) {
		p, err := pathArg(a, 0, "path_exists")
		if err != nil {
			return nil, err
		}
		_, statErr := os.Stat(p)
		return value.Bool(statErr == nil), nil
	})
	def("path_is_dir", 1, false, func(a []value.Value) (value.Value, error) {
		p, err := pathArg(a, 0, "path_is_dir")
		if err != nil {
			return nil, err
		}
		info, statErr := os.Stat(p)
		return value.Bool(statErr == nil && info.IsDir()), nil
	})
	def("mkdir_all", 1, false, func(a []value.Value) (value.Value, error) {
		p, err := pathArg(a, 0, "mkdir_all")
		if err != nil {
			return nil, err
		}
		// Every missing parent is created, and an existing directory is not an
		// error: a caller making somewhere to write into wants it to be there
		// afterwards, not to have been the one who made it.
		if mkErr := os.MkdirAll(p, 0o755); mkErr != nil {
			return fsErr(mkErr), nil
		}
		return fsOk(value.TheUnit), nil
	})
	def("remove_file", 1, false, func(a []value.Value) (value.Value, error) {
		p, err := pathArg(a, 0, "remove_file")
		if err != nil {
			return nil, err
		}
		info, statErr := os.Stat(p)
		if statErr == nil && info.IsDir() {
			return fsErr(fmt.Errorf("remove_file: %s is a directory (use remove_dir)", p)), nil
		}
		if rmErr := os.Remove(p); rmErr != nil {
			return fsErr(rmErr), nil
		}
		return fsOk(value.TheUnit), nil
	})
	// remove_dir removes a directory and everything under it. It is spelled
	// separately from remove_file, and refuses to be handed a file, because the
	// recursive one is the dangerous one and a caller should have to name it.
	def("remove_dir", 1, false, func(a []value.Value) (value.Value, error) {
		p, err := pathArg(a, 0, "remove_dir")
		if err != nil {
			return nil, err
		}
		info, statErr := os.Stat(p)
		if statErr != nil {
			return fsErr(statErr), nil
		}
		if !info.IsDir() {
			return fsErr(fmt.Errorf("remove_dir: %s is not a directory", p)), nil
		}
		if rmErr := os.RemoveAll(p); rmErr != nil {
			return fsErr(rmErr), nil
		}
		return fsOk(value.TheUnit), nil
	})
	def("rename_path", 2, false, func(a []value.Value) (value.Value, error) {
		from, err := pathArg(a, 0, "rename_path")
		if err != nil {
			return nil, err
		}
		to, err := pathArg(a, 1, "rename_path")
		if err != nil {
			return nil, err
		}
		if mvErr := os.Rename(from, to); mvErr != nil {
			return fsErr(mvErr), nil
		}
		return fsOk(value.TheUnit), nil
	})
	// temp_dir makes a fresh directory nobody else is using and hands back its
	// path. The caller removes it with remove_dir; nothing here removes it for
	// them, because a test that failed is usually a test whose output is worth
	// looking at.
	def("temp_dir", 1, false, func(a []value.Value) (value.Value, error) {
		prefix, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("temp_dir expects a name prefix")
		}
		dir, err := os.MkdirTemp("", prefix)
		if err != nil {
			return fsErr(err), nil
		}
		return fsOk(value.Str(dir)), nil
	})

	// --- paths --------------------------------------------------------------
	//
	// A path is text, and these are string operations on it: nothing here
	// touches the filesystem. The separator is `/` in what they produce, on
	// every platform, because a twill program's paths are written in its source
	// and a program that renders one differently on Windows is one that stores a
	// different manifest there. A `\` in what they are given is read as a
	// separator too, so a path handed in by the operating system still splits.

	def("path_join", -1, true, func(a []value.Value) (value.Value, error) {
		parts := make([]string, 0, len(a))
		for _, v := range a {
			s, ok := asStr(v)
			if !ok {
				return nil, fmt.Errorf("path_join expects strings")
			}
			if s != "" {
				parts = append(parts, s)
			}
		}
		return value.Str(cleanPath(strings.Join(parts, "/"))), nil
	})
	def("path_base", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("path_base expects a path")
		}
		return value.Str(pathBase(s)), nil
	})
	def("path_dir", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("path_dir expects a path")
		}
		return value.Str(pathDir(s)), nil
	})
	def("path_ext", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("path_ext expects a path")
		}
		base := pathBase(s)
		if i := strings.LastIndexByte(base, '.'); i > 0 {
			return value.Str(base[i:]), nil
		}
		return value.Str(""), nil
	})
	def("path_stem", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("path_stem expects a path")
		}
		base := pathBase(s)
		if i := strings.LastIndexByte(base, '.'); i > 0 {
			return value.Str(base[:i]), nil
		}
		return value.Str(base), nil
	})
	def("path_normalize", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("path_normalize expects a path")
		}
		return value.Str(cleanPath(s)), nil
	})
	def("path_is_abs", 1, false, func(a []value.Value) (value.Value, error) {
		s, ok := asStr(a[0])
		if !ok {
			return nil, fmt.Errorf("path_is_abs expects a path")
		}
		return value.Bool(filepath.IsAbs(s) || strings.HasPrefix(s, "/")), nil
	})
	def("cwd", 0, false, func(a []value.Value) (value.Value, error) {
		dir, err := os.Getwd()
		if err != nil {
			return fsErr(err), nil
		}
		return fsOk(value.Str(filepath.ToSlash(dir))), nil
	})
}

// cleanPath resolves `.` and `..` textually and renders the result with `/`
// separators. It is filepath.Clean with the answer forced to one spelling, so
// the same source produces the same path on every platform.
func cleanPath(p string) string {
	if p == "" {
		return "."
	}
	slashed := strings.ReplaceAll(p, `\`, "/")
	// A leading drive letter or UNC prefix is left as it was: it is not a
	// segment, and Clean's handling of it differs by platform.
	cleaned := filepath.ToSlash(filepath.Clean(slashed))
	return cleaned
}

func pathBase(p string) string {
	slashed := strings.ReplaceAll(p, `\`, "/")
	slashed = strings.TrimRight(slashed, "/")
	if slashed == "" {
		return "/"
	}
	if i := strings.LastIndexByte(slashed, '/'); i >= 0 {
		return slashed[i+1:]
	}
	return slashed
}

func pathDir(p string) string {
	slashed := strings.ReplaceAll(p, `\`, "/")
	slashed = strings.TrimRight(slashed, "/")
	i := strings.LastIndexByte(slashed, '/')
	switch {
	case i < 0:
		return "."
	case i == 0:
		return "/"
	default:
		return slashed[:i]
	}
}
