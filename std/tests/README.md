# std/tests

Two test suites live here, and only one of them runs today.

Run a suite by handing the file to the interpreter:

```
twill run std/tests/nn_test.tw
```

A passing suite prints one line per check and ends with `OK`. A failing one ends
with `FAILED` and exits non-zero, so a script can gate on it.

## Runs today

These are numeric-mode suites, written against the language the bootstrap
interpreter actually implements. They import `harness.tw` as a file.

| Suite | Module under test |
| --- | --- |
| `batch_test.tw` | `std/batch` |
| `frame_test.tw` | `std/frame` |
| `loss_test.tw` | `std/loss` |
| `metrics_test.tw` | `std/metrics` |
| `nn_test.tw` | `std/nn` |
| `optim_test.tw` | `std/optim` |

`harness.tw` is the numeric-mode harness, not a suite. Running it directly does
nothing and prints nothing.

## Written ahead of the language

These are `mode systems` suites, and the bootstrap interpreter rejects them at
parse time. That is expected. They were written alongside the systems-mode
modules they cover, and they are kept in the tree so the modules are not
shipping untested once the front end catches up. They are not a regression and
they are not skipped tests waiting on a bug fix.

| Suite | Module under test |
| --- | --- |
| `io_test.tw` | `std/io` |
| `json_test.tw` | `std/json` |
| `linalg_test.tw` | `std/linalg` |
| `random_test.tw` | `std/random` |
| `stats_test.tw` | `std/stats` |
| `text_test.tw` | `std/text` |
| `systems_harness.tw` | the harness the six above assert through |

Three constructs are what the parser stops on:

- **Typed annotations naming a generic type.** `fn av(xs: Arr[F64], v: F64)`
  fails with `expected ")" but found "["`. The parser has no bracketed type
  application.
- **Qualified type annotations.** `let rp: t.Report = t.new_report()` fails with
  `expected "=" but found "."`. An annotation cannot yet name a type through a
  module binding.
- **Struct field assignment.** `rp.passed = rp.passed + 1` fails with
  `unexpected token "="`. A field is readable but not yet an assignment target,
  which is what `docs/needs.md` NEEDS-42 asks for.

The same three constructs are why `go test ./internal/format` has six failing
subtests over `std/io.tw`, `std/json.tw`, `std/linalg.tw`, `std/random.tw`,
`std/stats.tw` and `std/text.tw`. The formatter shares the bootstrap parser, so
it cannot format what that parser cannot read. One cause, two symptoms.
