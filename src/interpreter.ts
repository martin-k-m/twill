// interpreter.ts — a tree-walking evaluator for Aster.

import { parse } from "./parser.ts";
import { Tensor, add, sub, mul, div, mod, neg, matmul, powScalar } from "./tensor.ts";
import {
  UNIT,
  bool,
  str,
  list,
  newEnv,
  envGet,
  envSet,
  truthy,
  isCallable,
  formatValue,
} from "./values.ts";
import type { Value, Env, Closure } from "./values.ts";
import { installBuiltins } from "./builtins.ts";
import type {
  Program,
  Stmt,
  Expr,
  Block,
} from "./ast.ts";

export class AsterRuntimeError extends Error {
  line: number;
  constructor(message: string, line: number) {
    super(`line ${line}: ${message}`);
    this.name = "AsterRuntimeError";
    this.line = line;
  }
}

// Control-flow signals threaded through statement evaluation.
type Flow =
  | { type: "normal"; value: Value }
  | { type: "return"; value: Value };

export interface RunOptions {
  print?: (s: string) => void;
}

export class Interpreter {
  global: Env;

  constructor(options: RunOptions = {}) {
    this.global = newEnv();
    const out = options.print ?? ((s: string) => console.log(s));
    installBuiltins(this.global, out, (callee, args) => this.apply(callee, args, 0));
  }

  run(source: string): Value {
    const program: Program = parse(source);
    let last: Value = UNIT;
    for (const stmt of program.body) {
      const flow = this.execStmt(stmt, this.global);
      if (flow.type === "return") return flow.value;
      last = flow.value;
    }
    return last;
  }

  // Convenience for the REPL: evaluate and return the final value.
  eval(source: string): Value {
    return this.run(source);
  }

  private execBlock(block: Block, env: Env): Flow {
    const scope = newEnv(env);
    let last: Value = UNIT;
    for (const stmt of block.body) {
      const flow = this.execStmt(stmt, scope);
      if (flow.type === "return") return flow;
      last = flow.value;
    }
    return { type: "normal", value: last };
  }

  private execStmt(stmt: Stmt, env: Env): Flow {
    switch (stmt.kind) {
      case "Let": {
        const value = this.evalExpr(stmt.value, env);
        env.vars.set(stmt.name, value);
        return { type: "normal", value: UNIT };
      }
      case "FnDecl": {
        const closure: Closure = {
          kind: "closure",
          params: stmt.params,
          body: stmt.body,
          env,
          name: stmt.name,
        };
        env.vars.set(stmt.name, closure);
        return { type: "normal", value: UNIT };
      }
      case "Assign": {
        const value = this.evalExpr(stmt.value, env);
        if (!envSet(env, stmt.name, value)) {
          throw new AsterRuntimeError(
            `cannot assign to undefined variable '${stmt.name}' (use 'let' first)`,
            stmt.line,
          );
        }
        return { type: "normal", value: UNIT };
      }
      case "While": {
        while (truthy(this.evalExpr(stmt.cond, env))) {
          const flow = this.execBlock(stmt.body, env);
          if (flow.type === "return") return flow;
        }
        return { type: "normal", value: UNIT };
      }
      case "For": {
        const iter = this.evalExpr(stmt.iter, env);
        const items = this.iterate(iter, stmt.line);
        for (const item of items) {
          const scope = newEnv(env);
          scope.vars.set(stmt.name, item);
          const flow = this.execBlockInScope(stmt.body, scope);
          if (flow.type === "return") return flow;
        }
        return { type: "normal", value: UNIT };
      }
      case "Return": {
        const value = stmt.value ? this.evalExpr(stmt.value, env) : UNIT;
        return { type: "return", value };
      }
      case "ExprStmt": {
        const value = this.evalExpr(stmt.expr, env);
        return { type: "normal", value };
      }
    }
  }

  private execBlockInScope(block: Block, scope: Env): Flow {
    let last: Value = UNIT;
    for (const stmt of block.body) {
      const flow = this.execStmt(stmt, scope);
      if (flow.type === "return") return flow;
      last = flow.value;
    }
    return { type: "normal", value: last };
  }

  private iterate(v: Value, line: number): Value[] {
    if (v instanceof Tensor) {
      if (v.shape.length === 1) {
        return Array.from(v.data, (x) => Tensor.scalar(x));
      }
      throw new AsterRuntimeError("can only iterate 1-D tensors", line);
    }
    if ((v as { kind?: string }).kind === "list") {
      return (v as { items: Value[] }).items;
    }
    throw new AsterRuntimeError("value is not iterable", line);
  }

  private evalExpr(expr: Expr, env: Env): Value {
    switch (expr.kind) {
      case "Number":
        return Tensor.scalar(expr.value);
      case "String":
        return str(expr.value);
      case "Bool":
        return bool(expr.value);
      case "Ident": {
        const v = envGet(env, expr.name);
        if (v === undefined) {
          throw new AsterRuntimeError(`undefined variable '${expr.name}'`, expr.line);
        }
        return v;
      }
      case "Tensor":
        return Tensor.fromNested(this.tensorNested(expr.elements, expr.line));
      case "List":
        return list(expr.elements.map((e) => this.evalExpr(e, env)));
      case "Lambda":
        return {
          kind: "closure",
          params: expr.params,
          body: expr.body,
          env,
          name: "",
        };
      case "Unary":
        return this.evalUnary(expr.op, this.evalExpr(expr.operand, env), expr.line);
      case "Binary":
        return this.evalBinary(expr, env);
      case "Call":
        return this.evalCall(expr, env);
      case "Index":
        return this.evalIndex(expr, env);
      case "If": {
        if (truthy(this.evalExpr(expr.cond, env))) {
          return this.execBlock(expr.then, env).value;
        }
        if (expr.else === null) return UNIT;
        if (expr.else.kind === "Block") return this.execBlock(expr.else, env).value;
        return this.evalExpr(expr.else, env);
      }
      case "Block":
        return this.execBlock(expr, env).value;
    }
  }

  private tensorNested(elements: Expr[], line: number): unknown {
    return elements.map((e) => {
      if (e.kind === "Number") return e.value;
      if (e.kind === "Unary" && e.op === "-" && e.operand.kind === "Number") {
        return -e.operand.value;
      }
      if (e.kind === "Tensor") return this.tensorNested(e.elements, line);
      throw new AsterRuntimeError("invalid element in tensor literal", line);
    });
  }

  private evalUnary(op: string, v: Value, line: number): Value {
    if (op === "-") {
      if (v instanceof Tensor) return neg(v);
      throw new AsterRuntimeError("unary '-' expects a number/tensor", line);
    }
    // logical not
    return bool(!truthy(v));
  }

  private evalBinary(expr: { op: string; left: Expr; right: Expr; line: number }, env: Env): Value {
    const op = expr.op;

    // Short-circuit logical operators.
    if (op === "and" || op === "&&") {
      const l = this.evalExpr(expr.left, env);
      return truthy(l) ? this.evalExpr(expr.right, env) : l;
    }
    if (op === "or" || op === "||") {
      const l = this.evalExpr(expr.left, env);
      return truthy(l) ? l : this.evalExpr(expr.right, env);
    }

    const l = this.evalExpr(expr.left, env);
    const r = this.evalExpr(expr.right, env);

    // Comparisons operate on scalar tensors.
    if (["==", "!=", "<", "<=", ">", ">="].includes(op)) {
      return bool(this.compare(op, l, r, expr.line));
    }

    if (!(l instanceof Tensor) || !(r instanceof Tensor)) {
      throw new AsterRuntimeError(`operator '${op}' expects numbers/tensors`, expr.line);
    }

    try {
      switch (op) {
        case "+":
          return add(l, r);
        case "-":
          return sub(l, r);
        case "*":
          return mul(l, r);
        case "/":
          return div(l, r);
        case "%":
          return mod(l, r);
        case "@":
          return matmul(l, r);
        case "^": {
          if (!r.isScalar) {
            throw new Error("exponent must be a scalar");
          }
          return powScalar(l, r.data[0]);
        }
        default:
          throw new Error(`unknown operator '${op}'`);
      }
    } catch (e) {
      throw new AsterRuntimeError((e as Error).message, expr.line);
    }
  }

  private compare(op: string, l: Value, r: Value, line: number): boolean {
    if (l instanceof Tensor && r instanceof Tensor && l.isScalar && r.isScalar) {
      const a = l.data[0];
      const b = r.data[0];
      switch (op) {
        case "==":
          return a === b;
        case "!=":
          return a !== b;
        case "<":
          return a < b;
        case "<=":
          return a <= b;
        case ">":
          return a > b;
        case ">=":
          return a >= b;
      }
    }
    if (op === "==" || op === "!=") {
      const eq = this.deepEqual(l, r);
      return op === "==" ? eq : !eq;
    }
    throw new AsterRuntimeError(`cannot compare these values with '${op}'`, line);
  }

  private deepEqual(l: Value, r: Value): boolean {
    if (l instanceof Tensor && r instanceof Tensor) {
      if (l.data.length !== r.data.length) return false;
      for (let i = 0; i < l.data.length; i++) if (l.data[i] !== r.data[i]) return false;
      return true;
    }
    const lk = (l as { kind?: string }).kind;
    const rk = (r as { kind?: string }).kind;
    if (lk !== rk) return false;
    if (lk === "bool") return (l as { value: boolean }).value === (r as { value: boolean }).value;
    if (lk === "str") return (l as { value: string }).value === (r as { value: string }).value;
    return l === r;
  }

  private evalCall(expr: { callee: Expr; args: Expr[]; line: number }, env: Env): Value {
    const callee = this.evalExpr(expr.callee, env);
    const args = expr.args.map((a) => this.evalExpr(a, env));
    return this.apply(callee, args, expr.line);
  }

  apply(callee: Value, args: Value[], line: number): Value {
    if (!isCallable(callee)) {
      throw new AsterRuntimeError(`value is not callable: ${formatValue(callee)}`, line);
    }
    if (callee.kind === "builtin") {
      if (callee.arity !== null && callee.arity !== args.length) {
        throw new AsterRuntimeError(
          `${callee.name} expects ${callee.arity} argument(s), got ${args.length}`,
          line,
        );
      }
      try {
        return callee.fn(args);
      } catch (e) {
        if (e instanceof AsterRuntimeError) throw e;
        throw new AsterRuntimeError((e as Error).message, line);
      }
    }
    // closure
    if (callee.params.length !== args.length) {
      throw new AsterRuntimeError(
        `${callee.name || "function"} expects ${callee.params.length} argument(s), got ${args.length}`,
        line,
      );
    }
    const scope = newEnv(callee.env);
    for (let i = 0; i < callee.params.length; i++) {
      scope.vars.set(callee.params[i], args[i]);
    }
    if (callee.body.kind === "Block") {
      return this.execBlockInScope(callee.body, scope).value;
    }
    return this.evalExpr(callee.body, scope);
  }

  private evalIndex(expr: { target: Expr; index: Expr; line: number }, env: Env): Value {
    const target = this.evalExpr(expr.target, env);
    const idxVal = this.evalExpr(expr.index, env);
    if (!(idxVal instanceof Tensor) || !idxVal.isScalar) {
      throw new AsterRuntimeError("index must be a scalar number", expr.line);
    }
    const idx = idxVal.data[0] | 0;

    if (target instanceof Tensor) {
      return this.indexTensor(target, idx, expr.line);
    }
    if ((target as { kind?: string }).kind === "list") {
      const items = (target as { items: Value[] }).items;
      if (idx < 0 || idx >= items.length) {
        throw new AsterRuntimeError(`list index ${idx} out of range`, expr.line);
      }
      return items[idx];
    }
    throw new AsterRuntimeError("value is not indexable", expr.line);
  }

  private indexTensor(t: Tensor, idx: number, line: number): Tensor {
    if (t.shape.length === 0) {
      throw new AsterRuntimeError("cannot index a scalar", line);
    }
    const dim0 = t.shape[0];
    if (idx < 0 || idx >= dim0) {
      throw new AsterRuntimeError(`tensor index ${idx} out of range [0, ${dim0})`, line);
    }
    if (t.shape.length === 1) {
      return Tensor.scalar(t.data[idx]);
    }
    // Return the idx-th slice along the first axis.
    const rest = t.shape.slice(1);
    let stride = 1;
    for (const d of rest) stride *= d;
    const slice = t.data.slice(idx * stride, (idx + 1) * stride);
    return new Tensor(Float64Array.from(slice), rest);
  }
}

export function run(source: string, options: RunOptions = {}): Value {
  return new Interpreter(options).run(source);
}
