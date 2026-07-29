// Tests for the differentiable tensor engine.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  Tensor,
  add,
  mul,
  sub,
  div,
  matmul,
  sum,
  relu,
  exp,
  tanh,
} from "../src/tensor.ts";

test("scalar add/mul forward", () => {
  const a = Tensor.scalar(3);
  const b = Tensor.scalar(4);
  assert.equal(add(a, b).data[0], 7);
  assert.equal(mul(a, b).data[0], 12);
});

test("gradient of x*x is 2x", () => {
  const x = new Tensor(Float64Array.of(5), [], true);
  const y = mul(x, x);
  y.backward();
  assert.equal(y.data[0], 25);
  assert.equal(x.grad![0], 10);
});

test("chain rule through div and sub", () => {
  // f(x) = (x - 1) / x ; f'(x) = 1/x^2
  const x = new Tensor(Float64Array.of(4), [], true);
  const f = div(sub(x, Tensor.scalar(1)), x);
  f.backward();
  assert.ok(Math.abs(f.data[0] - 0.75) < 1e-12);
  assert.ok(Math.abs(x.grad![0] - 1 / 16) < 1e-12);
});

test("vector sum gradient is ones", () => {
  const v = new Tensor(Float64Array.of(1, 2, 3), [3], true);
  const s = sum(v);
  s.backward();
  assert.equal(s.data[0], 6);
  assert.deepEqual(Array.from(v.grad!), [1, 1, 1]);
});

test("relu gradient gates negatives", () => {
  const v = new Tensor(Float64Array.of(-2, 0.5, 3), [3], true);
  const s = sum(relu(v));
  s.backward();
  assert.deepEqual(Array.from(v.grad!), [0, 1, 1]);
});

test("matmul forward and gradient", () => {
  // y = sum(A @ x), dy/dA = outer(ones, x), dy/dx = A^T @ ones
  const A = new Tensor(Float64Array.of(1, 2, 3, 4), [2, 2], true);
  const x = new Tensor(Float64Array.of(5, 6), [2], true);
  const y = sum(matmul(A, x));
  y.backward();
  // A @ x = [1*5+2*6, 3*5+4*6] = [17, 39]; sum = 56
  assert.equal(y.data[0], 56);
  // dy/dA row i = x  -> [[5,6],[5,6]]
  assert.deepEqual(Array.from(A.grad!), [5, 6, 5, 6]);
  // dy/dx = column sums of A = [1+3, 2+4] = [4, 6]
  assert.deepEqual(Array.from(x.grad!), [4, 6]);
});

test("exp and tanh derivatives", () => {
  const x = new Tensor(Float64Array.of(0.7), [], true);
  const y = exp(x);
  y.backward();
  assert.ok(Math.abs(x.grad![0] - Math.exp(0.7)) < 1e-12);

  const z = new Tensor(Float64Array.of(0.3), [], true);
  const t = tanh(z);
  t.backward();
  assert.ok(Math.abs(z.grad![0] - (1 - Math.tanh(0.3) ** 2)) < 1e-12);
});

test("gradient accumulates when a value is reused", () => {
  // f(x) = x + x = 2x ; f'(x) = 2
  const x = new Tensor(Float64Array.of(9), [], true);
  const y = add(x, x);
  y.backward();
  assert.equal(x.grad![0], 2);
});
