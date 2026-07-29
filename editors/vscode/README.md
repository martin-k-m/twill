# Raster for VS Code

Syntax highlighting for Raster (`.ra`) files: comments, strings, numbers,
keywords, operators, function definitions, and the built-in / autodiff
functions.

## Install locally

VS Code loads extensions from `~/.vscode/extensions`. To try this one without
packaging:

```bash
# from the repo root
cp -r editors/vscode ~/.vscode/extensions/raster-lang-0.1.0
```

Then reload VS Code. Any `.ra` file will be highlighted.

To build a `.vsix` package instead, install [`vsce`](https://github.com/microsoft/vscode-vsce)
and run `vsce package` inside `editors/vscode`.

## What's here

- `syntaxes/raster.tmLanguage.json` — the TextMate grammar.
- `language-configuration.json` — comments, brackets, auto-closing pairs.
- `package.json` — the extension manifest.
