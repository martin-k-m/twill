// ast.ts — the Aster abstract syntax tree.

export type Node =
  | Program
  | Stmt
  | Expr;

export interface Program {
  kind: "Program";
  body: Stmt[];
}

export type Stmt =
  | LetStmt
  | FnDecl
  | AssignStmt
  | WhileStmt
  | ForStmt
  | ReturnStmt
  | ExprStmt;

export interface LetStmt {
  kind: "Let";
  name: string;
  value: Expr;
  line: number;
}

export interface FnDecl {
  kind: "FnDecl";
  name: string;
  params: string[];
  body: Expr; // a Block or a single expression
  line: number;
}

export interface AssignStmt {
  kind: "Assign";
  name: string;
  value: Expr;
  line: number;
}

export interface WhileStmt {
  kind: "While";
  cond: Expr;
  body: Block;
  line: number;
}

export interface ForStmt {
  kind: "For";
  name: string;
  iter: Expr;
  body: Block;
  line: number;
}

export interface ReturnStmt {
  kind: "Return";
  value: Expr | null;
  line: number;
}

export interface ExprStmt {
  kind: "ExprStmt";
  expr: Expr;
  line: number;
}

export type Expr =
  | NumberLit
  | StringLit
  | BoolLit
  | Identifier
  | TensorLit
  | ListLit
  | Lambda
  | Unary
  | Binary
  | Call
  | Index
  | IfExpr
  | Block;

export interface NumberLit {
  kind: "Number";
  value: number;
  line: number;
}

export interface StringLit {
  kind: "String";
  value: string;
  line: number;
}

export interface BoolLit {
  kind: "Bool";
  value: boolean;
  line: number;
}

export interface Identifier {
  kind: "Ident";
  name: string;
  line: number;
}

export interface TensorLit {
  kind: "Tensor";
  elements: Expr[]; // numbers or nested TensorLit
  line: number;
}

export interface ListLit {
  kind: "List";
  elements: Expr[];
  line: number;
}

export interface Lambda {
  kind: "Lambda";
  params: string[];
  body: Expr;
  line: number;
}

export interface Unary {
  kind: "Unary";
  op: string;
  operand: Expr;
  line: number;
}

export interface Binary {
  kind: "Binary";
  op: string;
  left: Expr;
  right: Expr;
  line: number;
}

export interface Call {
  kind: "Call";
  callee: Expr;
  args: Expr[];
  line: number;
}

export interface Index {
  kind: "Index";
  target: Expr;
  index: Expr;
  line: number;
}

export interface IfExpr {
  kind: "If";
  cond: Expr;
  then: Block;
  else: Block | IfExpr | null;
  line: number;
}

export interface Block {
  kind: "Block";
  body: Stmt[];
  line: number;
}
