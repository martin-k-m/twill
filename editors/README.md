# Editors

Two separate things live here, and they are useful independently.

**Syntax highlighting** is `editors/vscode`, a published extension carrying a
TextMate grammar, snippets and the bracket/comment configuration. It needs
nothing installed beyond itself.

**`twill lsp`** is a language server, built into the `twill` binary as of 1.6.
It speaks JSON-RPC over stdin and stdout and gives an editor three things:

| | from |
| --- | --- |
| Diagnostics, republished as you type | the parser and `twill check` |
| Formatting | `twill fmt` |
| Hover: the inferred type and shape | the same answer as the REPL's `:type` and `:shape` |

There is no completion, deliberately. `docs/roadmap.md`'s own advice is not to
build one before the semantic information is reliable, and a list assembled from
a token scan would be a worse thing wearing a better thing's clothes.

Hover is the one worth setting up for. In a tensor-first language the question
you actually have is what shape something is, and hover answers it from the
checker without running anything -- so it costs nothing over
`logits @ w` even when running that would cost a gigabyte.

## Neovim

Built-in LSP, no plugin and no client code:

```lua
vim.filetype.add({ extension = { tw = "twill" } })

vim.api.nvim_create_autocmd("FileType", {
  pattern = "twill",
  callback = function(args)
    vim.lsp.start({
      name = "twill",
      cmd = { "twill", "lsp" },
      root_dir = vim.fs.dirname(vim.fs.find({ "spool.toml", ".git" }, { upward = true })[1]),
    }, { bufnr = args.buf })
  end,
})
```

`K` then shows the type and shape under the cursor, and `:lua vim.lsp.buf.format()`
formats the buffer.

## Helix

In `languages.toml`:

```toml
[language-server.twill]
command = "twill"
args = ["lsp"]

[[language]]
name = "twill"
scope = "source.twill"
file-types = ["tw"]
roots = ["spool.toml"]
comment-token = "#"
language-servers = ["twill"]
indent = { tab-width = 2, unit = "  " }
```

## VS Code

The extension in `editors/vscode` is grammar and snippets only: it does not
start the language server, because a VS Code client is a JavaScript extension
depending on `vscode-languageclient` from npm, and this repository has no
JavaScript build and no dependencies. Writing one is straightforward and is
worth doing; shipping one that has never been run is not, so it is not here.

Until then, any of the generic LSP bridge extensions will start `twill lsp` for
`.tw` files and give you the same three capabilities.

## Checking it works

The server is a program, so you can talk to it without an editor. It reads
`Content-Length`-framed JSON-RPC, the same as every language server:

```bash
printf 'Content-Length: 44\r\n\r\n{"jsonrpc":"2.0","id":1,"method":"initialize"}' | twill lsp
```

It answers with its capabilities. If your editor sees nothing, check that
`twill` is on its `PATH` -- `twill doctor` reports which binary that is, and a
stale one earlier on the path is the usual reason.
