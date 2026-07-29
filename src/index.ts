// index.ts — public library entry point for embedding Aster in Node/TypeScript.

export { Interpreter, run, AsterRuntimeError } from "./interpreter.ts";
export { parse } from "./parser.ts";
export { tokenize, AsterSyntaxError } from "./lexer.ts";
export { Tensor } from "./tensor.ts";
export { formatValue } from "./values.ts";
export type { Value } from "./values.ts";
