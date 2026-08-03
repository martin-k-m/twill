// Package std embeds Raster's standard library, the .ra files in this
// directory, so a single `raster` binary carries the library with it. An
// installed binary has no source tree to read them from, which is why they are
// compiled in rather than located at run time. The interpreter's RASTER_STD
// override is the one way to read the library from disk instead.
package std

import (
	"embed"
	"strings"
)

//go:embed *.ra
var sources embed.FS

// Read returns the source of module name ("nn", "optim", ...), reporting
// whether the module exists.
func Read(name string) (string, bool) {
	b, err := sources.ReadFile(name + ".ra")
	if err != nil {
		return "", false
	}
	return string(b), true
}

// Names lists the modules in the library, for error messages. embed.FS returns
// directory entries sorted by name, so the order is stable.
func Names() []string {
	entries, err := sources.ReadDir(".")
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		names = append(names, strings.TrimSuffix(e.Name(), ".ra"))
	}
	return names
}
