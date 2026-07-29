// builtins.ts — the standard library exposed to every Aster program.

import { Tensor, numel } from "./tensor.ts";
import * as ops from "./tensor.ts";
import {
  UNIT,
  str,
  list,
  formatValue,
} from "./values.ts";
import type { Value, Env, Builtin } from "./values.ts";

type Applier = (callee: Value, args: Value[]) => Value;

export function installBuiltins(env: Env, out: (s: string) => void, apply: Applier): void {
  const def = (name: string, arity: number | null, fn: (args: Value[]) => Value): void => {
    const b: Builtin = { kind: "builtin", name, arity, fn };
    env.vars.set(name, b);
  };

  // ---- I/O -------------------------------------------------------------
  def("print", null, (args) => {
    out(args.map(formatValue).join(" "));
    return UNIT;
  });

  // ---- unary math (differentiable) ------------------------------------
  const unaryOp = (name: string, f: (t: Tensor) => Tensor): void =>
    def(name, 1, (a) => f(asTensor(a[0], name)));

  unaryOp("relu", ops.relu);
  unaryOp("exp", ops.exp);
  unaryOp("log", ops.log);
  unaryOp("sin", ops.sin);
  unaryOp("cos", ops.cos);
  unaryOp("tanh", ops.tanh);
  unaryOp("sigmoid", ops.sigmoid);
  unaryOp("sqrt", ops.sqrt);
  unaryOp("sum", ops.sum);
  unaryOp("mean", ops.mean);

  def("abs", 1, (a) => {
    const t = asTensor(a[0], "abs");
    // |x| is relu(x) + relu(-x); keeps it differentiable.
    return ops.add(ops.relu(t), ops.relu(ops.neg(t)));
  });

  def("pow", 2, (a) => {
    const base = asTensor(a[0], "pow");
    const p = scalarOf(a[1], "pow");
    return ops.powScalar(base, p);
  });

  def("matmul", 2, (a) => ops.matmul(asTensor(a[0], "matmul"), asTensor(a[1], "matmul")));
  def("dot", 2, (a) => ops.matmul(asTensor(a[0], "dot"), asTensor(a[1], "dot")));

  // ---- automatic differentiation --------------------------------------
  // grad(f) -> a function returning df/d(arg0).
  def("grad", 1, (a) => {
    const f = a[0];
    return {
      kind: "builtin",
      name: "grad(fn)",
      arity: null,
      fn: (callArgs) => gradients(apply, f, callArgs).grad0,
    } satisfies Builtin;
  });

  // grads(f) -> a function returning the gradient w.r.t. every tensor arg,
  // as a list (constant tensors, matching each argument's shape).
  def("grads", 1, (a) => {
    const f = a[0];
    return {
      kind: "builtin",
      name: "grads(fn)",
      arity: null,
      fn: (callArgs) => list(gradients(apply, f, callArgs).all),
    } satisfies Builtin;
  });

  // value_and_grad(f) -> a function returning [value, grad0].
  def("value_and_grad", 1, (a) => {
    const f = a[0];
    return {
      kind: "builtin",
      name: "value_and_grad(fn)",
      arity: null,
      fn: (callArgs) => {
        const g = gradients(apply, f, callArgs);
        return list([g.value, g.grad0]);
      },
    } satisfies Builtin;
  });

  // ---- tensor construction --------------------------------------------
  def("tensor", 1, (a) => {
    const v = a[0];
    if (v instanceof Tensor) return v;
    return Tensor.fromNested(valueToNested(v));
  });

  def("scalar", 1, (a) => Tensor.scalar(scalarOf(a[0], "scalar")));

  def("zeros", null, (a) => Tensor.filled(shapeFromArgs(a, "zeros"), 0));
  def("ones", null, (a) => Tensor.filled(shapeFromArgs(a, "ones"), 1));
  def("fill", null, (a) => {
    if (a.length < 1) throw new Error("fill expects (value, ...shape)");
    const value = scalarOf(a[0], "fill");
    return Tensor.filled(shapeFromArgs(a.slice(1), "fill"), value);
  });

  def("randn", null, (a) => randomTensor(shapeFromArgs(a, "randn"), gaussian));
  def("rand", null, (a) => randomTensor(shapeFromArgs(a, "rand"), Math.random));

  def("eye", 1, (a) => {
    const n = intOf(a[0], "eye");
    const d = new Float64Array(n * n);
    for (let i = 0; i < n; i++) d[i * n + i] = 1;
    return new Tensor(d, [n, n]);
  });

  def("transpose", 1, (a) => {
    const t = asTensor(a[0], "transpose");
    if (t.shape.length !== 2) throw new Error("transpose expects a 2-D tensor");
    const [r, c] = t.shape;
    const d = new Float64Array(r * c);
    for (let i = 0; i < r; i++)
      for (let j = 0; j < c; j++) d[j * r + i] = t.data[i * c + j];
    return new Tensor(d, [c, r]);
  });

  // ---- inspection / utilities -----------------------------------------
  def("shape", 1, (a) => {
    const t = asTensor(a[0], "shape");
    return list(t.shape.map((d) => Tensor.scalar(d)));
  });

  def("len", 1, (a) => {
    const v = a[0];
    if (v instanceof Tensor) return Tensor.scalar(v.shape.length === 0 ? 1 : v.shape[0]);
    if ((v as { kind?: string }).kind === "list") {
      return Tensor.scalar((v as { items: Value[] }).items.length);
    }
    throw new Error("len expects a tensor or list");
  });

  def("item", 1, (a) => {
    const t = asTensor(a[0], "item");
    if (numel(t.shape) !== 1) throw new Error("item expects a single-element tensor");
    return Tensor.scalar(t.data[0]);
  });

  def("range", null, (a) => {
    let start = 0;
    let end: number;
    let step = 1;
    if (a.length === 1) {
      end = intOf(a[0], "range");
    } else if (a.length === 2) {
      start = intOf(a[0], "range");
      end = intOf(a[1], "range");
    } else if (a.length === 3) {
      start = intOf(a[0], "range");
      end = intOf(a[1], "range");
      step = intOf(a[2], "range");
    } else {
      throw new Error("range expects 1-3 arguments");
    }
    const items: Value[] = [];
    if (step === 0) throw new Error("range step cannot be 0");
    for (let x = start; step > 0 ? x < end : x > end; x += step) {
      items.push(Tensor.scalar(x));
    }
    return list(items);
  });

  def("list", null, (a) => list(a.slice()));

  def("str", 1, (a) => str(formatValue(a[0])));
}

// ---------------------------------------------------------------------------
// Autodiff core shared by grad / grads / value_and_grad
// ---------------------------------------------------------------------------

interface GradResult {
  value: Tensor;
  grad0: Tensor;
  all: Value[];
}

function gradients(apply: Applier, f: Value, callArgs: Value[]): GradResult {
  if (callArgs.length === 0) {
    throw new Error("gradient function requires at least one argument");
  }
  // Re-wrap tensor arguments as fresh, gradient-tracking leaves.
  const traced: (Tensor | null)[] = [];
  const passArgs: Value[] = callArgs.map((v) => {
    if (v instanceof Tensor) {
      const leaf = new Tensor(v.data.slice(), v.shape.slice(), true);
      traced.push(leaf);
      return leaf;
    }
    traced.push(null);
    return v;
  });

  const outVal = apply(f, passArgs);
  if (!(outVal instanceof Tensor) || !outVal.isScalar) {
    throw new Error("grad target must return a scalar");
  }
  outVal.backward();

  const gradOf = (leaf: Tensor | null): Tensor => {
    if (leaf === null) return Tensor.scalar(0);
    const g = leaf.grad ?? new Float64Array(leaf.data.length);
    return new Tensor(Float64Array.from(g), leaf.shape.slice());
  };

  return {
    value: Tensor.scalar(outVal.data[0]),
    grad0: gradOf(traced[0]),
    all: traced.map(gradOf),
  };
}

// ---------------------------------------------------------------------------
// Argument coercion helpers
// ---------------------------------------------------------------------------

function asTensor(v: Value, who: string): Tensor {
  if (v instanceof Tensor) return v;
  throw new Error(`${who} expects a tensor/number`);
}

function scalarOf(v: Value, who: string): number {
  const t = asTensor(v, who);
  if (numel(t.shape) !== 1) throw new Error(`${who} expects a scalar`);
  return t.data[0];
}

function intOf(v: Value, who: string): number {
  return Math.trunc(scalarOf(v, who));
}

function shapeFromArgs(args: Value[], who: string): number[] {
  if (args.length === 1) {
    const only = args[0];
    if ((only as { kind?: string }).kind === "list") {
      return (only as { items: Value[] }).items.map((x) => intOf(x, who));
    }
  }
  return args.map((x) => intOf(x, who));
}

function valueToNested(v: Value): unknown {
  if (v instanceof Tensor) return v.toNested();
  if ((v as { kind?: string }).kind === "list") {
    return (v as { items: Value[] }).items.map(valueToNested);
  }
  throw new Error("cannot convert value to a tensor");
}

function randomTensor(shape: number[], sample: () => number): Tensor {
  const d = new Float64Array(numel(shape));
  for (let i = 0; i < d.length; i++) d[i] = sample();
  return new Tensor(d, shape);
}

// Box–Muller transform for standard-normal samples.
function gaussian(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
