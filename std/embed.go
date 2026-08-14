// Package std embeds Twill's standard library, the .tw files in this
// directory, so a single `twill` binary carries the library with it. An
// installed binary has no source tree to read them from, which is why they are
// compiled in rather than located at run time. The interpreter's TWILL_STD
// override is the one way to read the library from disk instead.
package std

import (
	"embed"
	"path"
	"strings"
)

//go:embed *.tw term/*.tw
var sources embed.FS

// Read returns the source of module name ("nn", "optim", ...), reporting
// whether the module exists.
func Read(name string) (string, bool) {
	b, err := sources.ReadFile(name + ".tw")
	if err != nil {
		return "", false
	}
	return string(b), true
}

// Names lists the modules in the library, for error messages. The library is
// one level deep at most (the `term/` group), and a module in a group is named
// by its path, "term/caps", which is what an import writes. embed.FS returns
// directory entries sorted by name, so the order is stable.
func Names() []string {
	var names []string
	var walk func(dir string)
	walk = func(dir string) {
		entries, err := sources.ReadDir(path.Join(".", dir))
		if err != nil {
			return
		}
		for _, e := range entries {
			name := path.Join(dir, e.Name())
			if e.IsDir() {
				walk(name)
				continue
			}
			names = append(names, strings.TrimSuffix(name, ".tw"))
		}
	}
	walk("")
	return names
}
