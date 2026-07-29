// parser.ts — a recursive-descent / Pratt parser producing the Aster AST.

import { tokenize, AsterSyntaxError } from "./lexer.ts";
import type { Token } from "./lexer.ts";
import type {
  Program,
  Stmt,
  Expr,
  Block,
  IfExpr,
} from "./ast.ts";

// Binary operator precedence (higher binds tighter).
const PRECEDENCE: Record<string, number> = {
  or: 1,
  "||": 1,
  and: 2,
  "&&": 2,
  "==": 3,
  "!=": 3,
  "<": 4,
  "<=": 4,
  ">": 4,
  ">=": 4,
  "+": 5,
  "-": 5,
  "*": 6,
  "/": 6,
  "%": 6,
  "@": 6,
  "^": 7, // right-associative, handled specially
};

const RIGHT_ASSOC = new Set(["^"]);

export function parse(src: string): Program {
  return new Parser(tokenize(src)).parseProgram();
}

class Parser {
  tokens: Token[];
  pos: number;

  constructor(tokens: Token[]) {
    this.tokens = tokens;
    this.pos = 0;
  }

  private peek(o = 0): Token {
    return this.tokens[Math.min(this.pos + o, this.tokens.length - 1)];
  }

  private next(): Token {
    return this.tokens[this.pos++];
  }

  private atEnd(): boolean {
    return this.peek().type === "eof";
  }

  private check(value: string): boolean {
    const t = this.peek();
    return (t.type === "op" || t.type === "punct" || t.type === "keyword") &&
      t.value === value;
  }

  private match(value: string): boolean {
    if (this.check(value)) {
      this.next();
      return true;
    }
    return false;
  }

  private expect(value: string): Token {
    if (this.check(value)) return this.next();
    const t = this.peek();
    throw new AsterSyntaxError(
      `expected '${value}' but found '${t.value || t.type}'`,
      t.line,
      t.col,
    );
  }

  // Skip statement separators (newlines are insignificant; `;` is optional).
  private skipSeparators(): void {
    while (this.match(";")) {
      /* consume */
    }
  }

  parseProgram(): Program {
    const body: Stmt[] = [];
    this.skipSeparators();
    while (!this.atEnd()) {
      body.push(this.parseStmt());
      this.skipSeparators();
    }
    return { kind: "Program", body };
  }

  // -----------------------------------------------------------------------
  // Statements
  // -----------------------------------------------------------------------

  private parseStmt(): Stmt {
    const t = this.peek();

    if (t.type === "keyword") {
      switch (t.value) {
        case "let":
          return this.parseLet();
        case "fn":
          // `fn name(...)` is a declaration; `fn(...)` is a lambda expression.
          if (this.peek(1).type === "ident") return this.parseFnDecl();
          break;
        case "while":
          return this.parseWhile();
        case "for":
          return this.parseFor();
        case "return":
          return this.parseReturn();
      }
    }

    // Assignment: `ident = expr` (but not `==`).
    if (
      t.type === "ident" &&
      this.peek(1).type === "op" &&
      this.peek(1).value === "="
    ) {
      const name = this.next().value;
      this.next(); // '='
      const value = this.parseExpr();
      return { kind: "Assign", name, value, line: t.line };
    }

    const expr = this.parseExpr();
    return { kind: "ExprStmt", expr, line: t.line };
  }

  private parseLet(): Stmt {
    const line = this.next().line; // 'let'
    const name = this.expectIdent();
    this.expect("=");
    const value = this.parseExpr();
    return { kind: "Let", name, value, line };
  }

  private parseFnDecl(): Stmt {
    const line = this.next().line; // 'fn'
    const name = this.expectIdent();
    const params = this.parseParams();
    const body = this.parseFnBody();
    return { kind: "FnDecl", name, params, body, line };
  }

  private parseWhile(): Stmt {
    const line = this.next().line; // 'while'
    const cond = this.parseExpr();
    const body = this.parseBlock();
    return { kind: "While", cond, body, line };
  }

  private parseFor(): Stmt {
    const line = this.next().line; // 'for'
    const name = this.expectIdent();
    this.expect("in");
    const iter = this.parseExpr();
    const body = this.parseBlock();
    return { kind: "For", name, iter, body, line };
  }

  private parseReturn(): Stmt {
    const line = this.next().line; // 'return'
    // Bare `return` or `return expr`
    if (this.check("}") || this.check(";") || this.atEnd()) {
      return { kind: "Return", value: null, line };
    }
    const value = this.parseExpr();
    return { kind: "Return", value, line };
  }

  // -----------------------------------------------------------------------
  // Expressions (Pratt)
  // -----------------------------------------------------------------------

  private parseExpr(): Expr {
    return this.parseBinary(0);
  }

  private parseBinary(minPrec: number): Expr {
    let left = this.parseUnary();
    for (;;) {
      const t = this.peek();
      const op = t.value;
      const isOp =
        (t.type === "op" || (t.type === "keyword" && (op === "and" || op === "or")));
      if (!isOp) break;
      const prec = PRECEDENCE[op];
      if (prec === undefined || prec < minPrec) break;
      this.next();
      const nextMin = RIGHT_ASSOC.has(op) ? prec : prec + 1;
      const right = this.parseBinary(nextMin);
      left = { kind: "Binary", op, left, right, line: t.line };
    }
    return left;
  }

  private parseUnary(): Expr {
    const t = this.peek();
    if (
      (t.type === "op" && (t.value === "-" || t.value === "!")) ||
      (t.type === "keyword" && t.value === "not")
    ) {
      this.next();
      const operand = this.parseUnary();
      return { kind: "Unary", op: t.value, operand, line: t.line };
    }
    return this.parsePostfix();
  }

  private parsePostfix(): Expr {
    let expr = this.parsePrimary();
    for (;;) {
      if (this.check("(")) {
        const line = this.peek().line;
        const args = this.parseArgs();
        expr = { kind: "Call", callee: expr, args, line };
      } else if (this.check("[")) {
        const line = this.next().line; // '['
        const index = this.parseExpr();
        this.expect("]");
        expr = { kind: "Index", target: expr, index, line };
      } else {
        break;
      }
    }
    return expr;
  }

  private parsePrimary(): Expr {
    const t = this.peek();

    if (t.type === "number") {
      this.next();
      return { kind: "Number", value: Number(t.value), line: t.line };
    }
    if (t.type === "string") {
      this.next();
      return { kind: "String", value: t.value, line: t.line };
    }
    if (t.type === "keyword") {
      switch (t.value) {
        case "true":
          this.next();
          return { kind: "Bool", value: true, line: t.line };
        case "false":
          this.next();
          return { kind: "Bool", value: false, line: t.line };
        case "if":
          return this.parseIf();
        case "fn":
          return this.parseLambda();
      }
    }
    if (t.type === "ident") {
      this.next();
      return { kind: "Ident", name: t.value, line: t.line };
    }
    if (this.check("(")) {
      this.next();
      const inner = this.parseExpr();
      this.expect(")");
      return inner;
    }
    if (this.check("[")) {
      return this.parseTensorOrList();
    }
    if (this.check("{")) {
      return this.parseBlock();
    }

    throw new AsterSyntaxError(
      `unexpected token '${t.value || t.type}'`,
      t.line,
      t.col,
    );
  }

  private parseIf(): IfExpr {
    const line = this.next().line; // 'if'
    const cond = this.parseExpr();
    const then = this.parseBlock();
    let elseBranch: Block | IfExpr | null = null;
    if (this.match("else")) {
      if (this.check("if")) elseBranch = this.parseIf();
      else elseBranch = this.parseBlock();
    }
    return { kind: "If", cond, then, else: elseBranch, line };
  }

  private parseLambda(): Expr {
    const line = this.next().line; // 'fn'
    const params = this.parseParams();
    const body = this.parseFnBody();
    return { kind: "Lambda", params, body, line };
  }

  // A bracketed literal is a tensor when every element is a number/nested
  // tensor, otherwise a general list.
  private parseTensorOrList(): Expr {
    const line = this.next().line; // '['
    const elements: Expr[] = [];
    if (!this.check("]")) {
      elements.push(this.parseExpr());
      while (this.match(",")) {
        if (this.check("]")) break; // trailing comma
        elements.push(this.parseExpr());
      }
    }
    this.expect("]");
    const isTensor = elements.every(
      (e) =>
        e.kind === "Number" ||
        e.kind === "Tensor" ||
        (e.kind === "Unary" && e.op === "-" && e.operand.kind === "Number"),
    );
    if (isTensor && elements.length > 0) {
      return { kind: "Tensor", elements, line };
    }
    return { kind: "List", elements, line };
  }

  private parseBlock(): Block {
    const line = this.expect("{").line;
    const body: Stmt[] = [];
    this.skipSeparators();
    while (!this.check("}") && !this.atEnd()) {
      body.push(this.parseStmt());
      this.skipSeparators();
    }
    this.expect("}");
    return { kind: "Block", body, line };
  }

  private parseFnBody(): Expr {
    if (this.match("=")) return this.parseExpr();
    return this.parseBlock();
  }

  private parseParams(): string[] {
    this.expect("(");
    const params: string[] = [];
    if (!this.check(")")) {
      params.push(this.expectIdent());
      while (this.match(",")) {
        if (this.check(")")) break;
        params.push(this.expectIdent());
      }
    }
    this.expect(")");
    return params;
  }

  private parseArgs(): Expr[] {
    this.expect("(");
    const args: Expr[] = [];
    if (!this.check(")")) {
      args.push(this.parseExpr());
      while (this.match(",")) {
        if (this.check(")")) break;
        args.push(this.parseExpr());
      }
    }
    this.expect(")");
    return args;
  }

  private expectIdent(): string {
    const t = this.peek();
    if (t.type === "ident") {
      this.next();
      return t.value;
    }
    throw new AsterSyntaxError(
      `expected identifier but found '${t.value || t.type}'`,
      t.line,
      t.col,
    );
  }
}
