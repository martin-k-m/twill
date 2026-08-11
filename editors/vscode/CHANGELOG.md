# Change Log

## 1.0.0

First Marketplace release.

- Rewrote the TextMate grammar to cover current Twill syntax: the `mode`
  declaration, `enum`/`struct`/`type`/`unit` declarations, `match`, `break`/
  `continue`, imports (`import ... as`), logical and bitwise operators, built-in
  types and dtype names, variant constructors (`Ok`/`Err`/`Some`/`None`), the
  full built-in / autodiff function set, hex and underscored numeric literals,
  and the `->`, `=>` and postfix `?` operators.
- Added code snippets (`fn`, `let`, `for`, `while`, `if`, `ifelse`, `match`,
  `enum`, `struct`, `import`, `mode`, `grad`).
- Added indentation rules, `# region` folding, and comment-continuation on Enter.

## 0.1.0

- Initial syntax highlighting for `.tw` files.
