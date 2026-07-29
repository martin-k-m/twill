// lexer.ts — turns Aster source text into a flat token stream.

export type TokenType =
  | "number"
  | "string"
  | "ident"
  | "keyword"
  | "op"
  | "punct"
  | "eof";

export interface Token {
  type: TokenType;
  value: string;
  line: number;
  col: number;
}

const KEYWORDS = new Set([
  "let",
  "fn",
  "if",
  "else",
  "while",
  "for",
  "in",
  "return",
  "true",
  "false",
  "and",
  "or",
  "not",
]);

// Multi-character operators, longest first so the scanner is greedy.
const MULTI_OPS = ["==", "!=", "<=", ">=", "->", "&&", "||"];

export class AsterSyntaxError extends Error {
  line: number;
  col: number;
  constructor(message: string, line: number, col: number) {
    super(`line ${line}:${col}: ${message}`);
    this.name = "AsterSyntaxError";
    this.line = line;
    this.col = col;
  }
}

export function tokenize(src: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  let line = 1;
  let col = 1;

  const peek = (o = 0) => src[i + o];
  const advance = () => {
    const ch = src[i++];
    if (ch === "\n") {
      line++;
      col = 1;
    } else {
      col++;
    }
    return ch;
  };

  while (i < src.length) {
    const ch = peek();

    // Whitespace
    if (ch === " " || ch === "\t" || ch === "\r" || ch === "\n") {
      advance();
      continue;
    }

    // Comments: `# ...` to end of line
    if (ch === "#") {
      while (i < src.length && peek() !== "\n") advance();
      continue;
    }

    const startLine = line;
    const startCol = col;

    // Numbers (integers, floats, scientific notation)
    if (isDigit(ch) || (ch === "." && isDigit(peek(1)))) {
      let num = "";
      while (i < src.length && isDigit(peek())) num += advance();
      if (peek() === ".") {
        num += advance();
        while (i < src.length && isDigit(peek())) num += advance();
      }
      if (peek() === "e" || peek() === "E") {
        num += advance();
        if (peek() === "+" || peek() === "-") num += advance();
        while (i < src.length && isDigit(peek())) num += advance();
      }
      tokens.push({ type: "number", value: num, line: startLine, col: startCol });
      continue;
    }

    // Identifiers / keywords
    if (isIdentStart(ch)) {
      let name = "";
      while (i < src.length && isIdentPart(peek())) name += advance();
      const type: TokenType = KEYWORDS.has(name) ? "keyword" : "ident";
      tokens.push({ type, value: name, line: startLine, col: startCol });
      continue;
    }

    // Strings
    if (ch === '"') {
      advance();
      let str = "";
      while (i < src.length && peek() !== '"') {
        const c = advance();
        if (c === "\\") {
          const esc = advance();
          str += unescape(esc);
        } else {
          str += c;
        }
      }
      if (i >= src.length) {
        throw new AsterSyntaxError("unterminated string", startLine, startCol);
      }
      advance(); // closing quote
      tokens.push({ type: "string", value: str, line: startLine, col: startCol });
      continue;
    }

    // Multi-character operators
    const two = ch + (peek(1) ?? "");
    if (MULTI_OPS.includes(two)) {
      advance();
      advance();
      tokens.push({ type: "op", value: two, line: startLine, col: startCol });
      continue;
    }

    // Single-character operators and punctuation
    if ("+-*/%@^<>=!".includes(ch)) {
      advance();
      tokens.push({ type: "op", value: ch, line: startLine, col: startCol });
      continue;
    }
    if ("()[]{},:;".includes(ch)) {
      advance();
      tokens.push({ type: "punct", value: ch, line: startLine, col: startCol });
      continue;
    }

    throw new AsterSyntaxError(`unexpected character '${ch}'`, startLine, startCol);
  }

  tokens.push({ type: "eof", value: "", line, col });
  return tokens;
}

function isDigit(ch: string | undefined): boolean {
  return ch !== undefined && ch >= "0" && ch <= "9";
}

function isIdentStart(ch: string | undefined): boolean {
  return ch !== undefined && (/[A-Za-z_]/.test(ch));
}

function isIdentPart(ch: string | undefined): boolean {
  return ch !== undefined && (/[A-Za-z0-9_]/.test(ch));
}

function unescape(ch: string): string {
  switch (ch) {
    case "n":
      return "\n";
    case "t":
      return "\t";
    case "r":
      return "\r";
    case "\\":
      return "\\";
    case '"':
      return '"';
    default:
      return ch;
  }
}
