// Command extract pulls the .ra sources embedded in the Go test files out into
// standalone fixtures under testdata/.
//
// The fixtures are the specification, and until now they existed only as string
// literals that only a Go test binary could reach. Extracting them mechanically
// rather than by hand is the only way the corpus stays honest as the tests
// change: rerun this and the corpus is regenerated from the same source of
// truth. The Go tests are left exactly as they are.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	rparser "github.com/martin-k-m/raster/internal/parser"
)

// unportable lists fixtures whose recorded output is not a fact about Raster.
// Each of these opens a file that is not there, so its golden ends in the
// operating system's phrasing ("no such file or directory" against "The system
// cannot find the file specified") and in that platform's path separator. A
// golden nobody can reproduce on another machine is worse than no golden, so
// they are excluded rather than tolerated. Keyed by the same content hash the
// fixture filenames use.
var unportable = map[string]bool{
	"207dc3dd7cad5f41": true, // load("x.npy")
	"1b31aaa6288bd9b4": true, // read_frame("d.csv")
	"5108353b42442571": true, // read_csv("data.csv")
}

func main() {
	out := flag.String("out", "testdata/cases", "directory to write fixtures into")
	flag.Parse()
	roots := flag.Args()
	if len(roots) == 0 {
		roots = []string{"internal/interp", "internal/checker"}
	}

	seen := map[string]bool{}
	type fixture struct {
		name string
		src  string
	}
	var fixtures []fixture

	for _, root := range roots {
		files, err := filepath.Glob(filepath.Join(root, "*_test.go"))
		if err != nil {
			fatal(err)
		}
		sort.Strings(files)
		for _, file := range files {
			for _, lit := range rasterLiterals(file) {
				sum := sha256.Sum256([]byte(lit))
				key := hex.EncodeToString(sum[:8])
				if seen[key] || unportable[key] {
					continue
				}
				seen[key] = true
				base := strings.TrimSuffix(filepath.Base(file), "_test.go")
				pkg := filepath.Base(root)
				fixtures = append(fixtures, fixture{
					name: fmt.Sprintf("%s_%s_%s.ra", pkg, base, key),
					src:  lit,
				})
			}
		}
	}

	if err := os.MkdirAll(*out, 0o755); err != nil {
		fatal(err)
	}
	for _, f := range fixtures {
		body := f.src
		if !strings.HasSuffix(body, "\n") {
			body += "\n"
		}
		if err := os.WriteFile(filepath.Join(*out, f.name), []byte(body), 0o644); err != nil {
			fatal(err)
		}
	}
	fmt.Printf("wrote %d fixtures to %s\n", len(fixtures), *out)
}

// rasterLiterals returns every string literal in a Go test file that is
// plausibly a Raster program.
func rasterLiterals(path string) []string {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		fatal(err)
	}
	var found []string
	ast.Inspect(file, func(n ast.Node) bool {
		lit, ok := n.(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return true
		}
		s, err := strconv.Unquote(lit.Value)
		if err != nil {
			return true
		}
		if src, ok := asRasterSource(s); ok {
			found = append(found, src)
		}
		return true
	})
	return found
}

// asRasterSource decides whether a Go string literal is a Raster program.
//
// Parsing it is necessary but nowhere near sufficient: a test's expected-error
// substring like "inner 2 != 1" or "shape mismatch" parses cleanly as an
// expression, and a corpus full of those would be noise that still had to be
// diffed. So a literal also has to look like code, by containing a statement
// keyword or a call or an operator. The residual risk is the other direction,
// dropping a one-token fixture such as "1 + 2", which is covered many times
// over by the fixtures that are kept.
func asRasterSource(s string) (string, bool) {
	trimmed := dedent(s)
	if strings.TrimSpace(trimmed) == "" {
		return "", false
	}
	if len(trimmed) > 8000 {
		return "", false
	}
	if _, err := rparser.Parse(trimmed); err != nil {
		return "", false
	}
	if !looksLikeCode(trimmed) {
		return "", false
	}
	return trimmed, true
}

func looksLikeCode(s string) bool {
	for _, kw := range []string{"let ", "fn ", "for ", "while ", "if ", "import ", "type ", "return ", "print(", "(", "[", "@", "+", "-", "*", "/", "^", "="} {
		if strings.Contains(s, kw) {
			return true
		}
	}
	return false
}

// dedent strips the common leading tabs the tests use for raw-string blocks.
// Raster is not indentation sensitive, but leaving the indentation in makes the
// fixtures unreadable and makes fmt idempotence checks report the whole file.
func dedent(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	lines := strings.Split(s, "\n")
	prefix := -1
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		n := len(line) - len(strings.TrimLeft(line, "\t "))
		if prefix < 0 || n < prefix {
			prefix = n
		}
	}
	if prefix <= 0 {
		return strings.TrimLeft(s, "\n")
	}
	for i, line := range lines {
		if len(line) >= prefix {
			lines[i] = line[prefix:]
		} else {
			lines[i] = strings.TrimLeft(line, "\t ")
		}
	}
	return strings.TrimLeft(strings.Join(lines, "\n"), "\n")
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "extract:", err)
	os.Exit(1)
}
