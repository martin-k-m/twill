// values.ts — the runtime value model shared by the interpreter and builtins.

import { Tensor } from "./tensor.ts";
import type { Expr } from "./ast.ts";

export interface Env {
  vars: Map<string, Value>;
  parent: Env | null;
}

export interface Closure {
  kind: "closure";
  params: string[];
  body: Expr;
  env: Env;
  name: string;
}

export interface Builtin {
  kind: "builtin";
  name: string;
  arity: number | null; // null = variadic
  fn: (args: Value[]) => Value;
}

export interface BoolVal {
  kind: "bool";
  value: boolean;
}

export interface StrVal {
  kind: "str";
  value: string;
}

export interface ListVal {
  kind: "list";
  items: Value[];
}

export interface UnitVal {
  kind: "unit";
}

export type Value =
  | Tensor
  | Closure
  | Builtin
  | BoolVal
  | StrVal
  | ListVal
  | UnitVal;

export const UNIT: UnitVal = { kind: "unit" };

export function isTensor(v: Value): v is Tensor {
  return v instanceof Tensor;
}

export function isCallable(v: Value): v is Closure | Builtin {
  return (v as { kind?: string }).kind === "closure" ||
    (v as { kind?: string }).kind === "builtin";
}

export function bool(b: boolean): BoolVal {
  return { kind: "bool", value: b };
}

export function str(s: string): StrVal {
  return { kind: "str", value: s };
}

export function list(items: Value[]): ListVal {
  return { kind: "list", items };
}

export function newEnv(parent: Env | null = null): Env {
  return { vars: new Map(), parent };
}

export function envGet(env: Env, name: string): Value | undefined {
  let e: Env | null = env;
  while (e) {
    const v = e.vars.get(name);
    if (v !== undefined) return v;
    e = e.parent;
  }
  return undefined;
}

export function envSet(env: Env, name: string, value: Value): boolean {
  let e: Env | null = env;
  while (e) {
    if (e.vars.has(name)) {
      e.vars.set(name, value);
      return true;
    }
    e = e.parent;
  }
  return false;
}

export function truthy(v: Value): boolean {
  if (v instanceof Tensor) {
    if (v.isScalar) return v.data[0] !== 0;
    return v.data.length > 0;
  }
  switch (v.kind) {
    case "bool":
      return v.value;
    case "str":
      return v.value.length > 0;
    case "list":
      return v.items.length > 0;
    case "unit":
      return false;
    default:
      return true;
  }
}

// Human-readable formatting used by `print` and the REPL.
export function formatValue(v: Value): string {
  if (v instanceof Tensor) {
    if (v.isScalar) return formatNumber(v.data[0]);
    return `tensor(${formatNested(v.toNested())}, shape=[${v.shape.join(", ")}])`;
  }
  switch (v.kind) {
    case "bool":
      return v.value ? "true" : "false";
    case "str":
      return v.value;
    case "list":
      return `[${v.items.map(formatValue).join(", ")}]`;
    case "closure":
      return `<fn ${v.name || "anonymous"}(${v.params.join(", ")})>`;
    case "builtin":
      return `<builtin ${v.name}>`;
    case "unit":
      return "()";
  }
}

function formatNested(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(formatNested).join(", ")}]`;
  return formatNumber(value as number);
}

export function formatNumber(n: number): string {
  if (Number.isInteger(n)) return n.toString();
  // Trim to a readable precision without trailing noise.
  return Number(n.toFixed(6)).toString();
}
