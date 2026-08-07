# Contributing

Twill is an early-stage language. Bug reports, small fixes, and design
discussion are all welcome.

## Building and testing

You need Go 1.23 or newer.

```bash
go build -o twill ./cmd/twill   # build the binary
go test ./...                     # run the tests
go vet ./...                      # static checks
gofmt -l .                        # should print nothing
```

Or use the Makefile: `make build`, `make test`, `make check`, `make bench`.

## Layout

```
cmd/twill/          the command (run / check / repl)
internal/lexer/      source text -> tokens
internal/parser/     tokens -> AST
internal/ast/        AST node types
internal/tensor/     the differentiable tensor engine
internal/value/      runtime values and environments
internal/interp/     the tree-walking interpreter + builtins
internal/checker/    static shape analysis
std/                 libraries written in Twill (nn.tw, optim.tw)
examples/            runnable .tw programs
editors/vscode/      syntax highlighting
```

## Conventions

- Run `gofmt` before committing; CI checks formatting and `go vet`.
- Every tensor operation that participates in autodiff should have a
  gradient-check test in `internal/tensor/gradcheck_test.go` (compare the
  analytic gradient to a finite-difference estimate).
- The shape checker must stay conservative: only report a diagnostic when a
  mismatch is certain. If in doubt, return an unknown shape rather than guess.
- Keep the language small. New builtins are cheap; new syntax should earn its
  place.

## Adding a builtin

1. Implement the operation in `internal/tensor` (with a backward pass if it's
   differentiable) and add a gradient-check test.
2. Register it in `internal/interp/builtins.go`.
3. Teach the shape checker its result shape in `internal/checker/checker.go`
   and add its name to `builtinNames`.
4. Document it in `docs/language-guide.md`.
