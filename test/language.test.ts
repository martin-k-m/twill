// End-to-end tests running Aster source through the interpreter.

import { test } from "node:test";
import assert from "node:assert/strict";
import { Interpreter } from "../src/interpreter.ts";
import { Tensor } from "../src/tensor.ts";
import { formatValue } from "../src/values.ts";
import type { Value } from "../src/values.ts";

// Run a program, returning both its final value and captured stdout lines.
function run(src: string): { value: Value; out: string[] } {
  const out: string[] = [];
  const interp = new Interpreter({ print: (s) => out.push(s) });
  const value = interp.run(src);
  return { value, out };
}

function scalar(src: string): number {
  const { value } = run(src);
  assert.ok(value instanceof Tensor, "expected a tensor result");
  return (value as Tensor).data[0];
}

test("arithmetic and precedence", () => {
  assert.equal(scalar("1 + 2 * 3"), 7);
  assert.equal(scalar("(1 + 2) * 3"), 9);
  assert.equal(scalar("2 ^ 10"), 1024);
  assert.equal(scalar("17 % 5"), 2);
  assert.equal(scalar("-3 + 4"), 1);
});

test("let bindings and functions", () => {
  const src = `
    let a = 10
    fn double(x) = x * 2
    double(a) + 1
  `;
  assert.equal(scalar(src), 21);
});

test("closures capture their environment", () => {
  const src = `
    fn adder(n) = fn(x) = x + n
    let add5 = adder(5)
    add5(37)
  `;
  assert.equal(scalar(src), 42);
});

test("if / else as an expression", () => {
  assert.equal(scalar("if 3 > 2 { 100 } else { 200 }"), 100);
  assert.equal(scalar("if 1 > 2 { 100 } else { 200 }"), 200);
});

test("while loop with reassignment", () => {
  const src = `
    let i = 0
    let total = 0
    while i < 5 {
      total = total + i
      i = i + 1
    }
    total
  `;
  assert.equal(scalar(src), 10);
});

test("for loop over range", () => {
  const src = `
    let s = 0
    for k in range(1, 6) { s = s + k }
    s
  `;
  assert.equal(scalar(src), 15);
});

test("tensor literals, matmul, and indexing", () => {
  assert.equal(scalar("[1.0, 2.0, 3.0] @ [4.0, 5.0, 6.0]"), 32);
  assert.equal(scalar("let m = [[1.0, 2.0], [3.0, 4.0]] \n m[1][0]"), 3);
  assert.equal(scalar("sum([1.0, 2.0, 3.0, 4.0])"), 10);
});

test("grad builtin computes derivatives", () => {
  assert.equal(scalar("grad(fn(x) = x * x * x)(2.0)"), 12);
  assert.equal(scalar("grad(fn(x) = sum(x * x))([3.0, 4.0])[0]"), 6);
});

test("grads returns per-argument gradients", () => {
  const src = `
    fn bil(a, b) = sum(a * b)
    let g = grads(bil)([1.0, 2.0], [10.0, 20.0])
    g[0][1]
  `;
  assert.equal(scalar(src), 20);
});

test("gradient descent reduces a quadratic loss", () => {
  // Minimize (w - 3)^2 starting from 0.
  const src = `
    let w = 0.0
    fn loss(w) = (w - 3.0) * (w - 3.0)
    for step in range(200) {
      let g = grad(loss)(w)
      w = w - g * 0.1
    }
    w
  `;
  assert.ok(Math.abs(scalar(src) - 3) < 1e-3);
});

test("print formatting", () => {
  const { out } = run(`print("v =", [1.0, 2.0])`);
  assert.equal(out[0], "v = tensor([1, 2], shape=[2])");
});

test("shape builtin", () => {
  const { value } = run("shape([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])");
  assert.equal(formatValue(value), "[2, 3]");
});
