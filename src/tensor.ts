// tensor.ts — the differentiable tensor engine at the core of Aster.
//
// Every numeric value in Aster is a Tensor. Scalars are rank-0 tensors
// (shape []). Operations build a lazy reverse-mode autodiff graph, but only
// when at least one input has `requiresGrad` set — so ordinary evaluation
// stays cheap and `grad(...)` is exact.

export type Shape = number[];

export function numel(shape: Shape): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

export function shapeEqual(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export class Tensor {
  data: Float64Array;
  shape: Shape;
  requiresGrad: boolean;
  grad: Float64Array | null;
  // Autodiff graph links (populated only inside a traced computation).
  _prev: Tensor[];
  _backward: () => void;

  constructor(data: Float64Array, shape: Shape, requiresGrad = false) {
    this.data = data;
    this.shape = shape;
    this.requiresGrad = requiresGrad;
    this.grad = null;
    this._prev = [];
    this._backward = () => {};
  }

  get size(): number {
    return numel(this.shape);
  }

  get isScalar(): boolean {
    return this.shape.length === 0;
  }

  static scalar(x: number, requiresGrad = false): Tensor {
    return new Tensor(Float64Array.of(x), [], requiresGrad);
  }

  static filled(shape: Shape, value: number): Tensor {
    const d = new Float64Array(numel(shape));
    d.fill(value);
    return new Tensor(d, shape.slice());
  }

  // Build a tensor from an arbitrarily nested JS array of numbers.
  static fromNested(value: unknown): Tensor {
    const shape: Shape = [];
    let probe: unknown = value;
    while (Array.isArray(probe)) {
      shape.push(probe.length);
      probe = probe[0];
    }
    const flat: number[] = [];
    const walk = (v: unknown, depth: number): void => {
      if (Array.isArray(v)) {
        if (v.length !== shape[depth]) {
          throw new Error("ragged tensor literal: inconsistent row lengths");
        }
        for (const item of v) walk(item, depth + 1);
      } else if (typeof v === "number") {
        flat.push(v);
      } else {
        throw new Error("tensor literals may only contain numbers");
      }
    };
    walk(value, 0);
    return new Tensor(Float64Array.from(flat), shape);
  }

  toNested(): unknown {
    if (this.isScalar) return this.data[0];
    const build = (offset: number, dim: number): [unknown, number] => {
      if (dim === this.shape.length - 1) {
        const row: number[] = [];
        for (let i = 0; i < this.shape[dim]; i++) row.push(this.data[offset + i]);
        return [row, offset + this.shape[dim]];
      }
      const rows: unknown[] = [];
      let off = offset;
      for (let i = 0; i < this.shape[dim]; i++) {
        const [r, next] = build(off, dim + 1);
        rows.push(r);
        off = next;
      }
      return [rows, off];
    };
    return build(0, 0)[0];
  }

  ensureGrad(): Float64Array {
    if (this.grad === null) this.grad = new Float64Array(this.data.length);
    return this.grad;
  }

  // Reverse-mode backpropagation. Only valid from a scalar output.
  backward(): void {
    if (!this.isScalar) {
      throw new Error("backward() may only be called on a scalar output");
    }
    const topo: Tensor[] = [];
    const seen = new Set<Tensor>();
    const visit = (t: Tensor): void => {
      if (seen.has(t)) return;
      seen.add(t);
      for (const p of t._prev) visit(p);
      topo.push(t);
    };
    visit(this);
    this.ensureGrad()[0] = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

// ---------------------------------------------------------------------------
// Operation helpers
// ---------------------------------------------------------------------------

function tracked(out: Tensor, prev: Tensor[], backward: () => void): Tensor {
  const rg = prev.some((p) => p.requiresGrad);
  if (rg) {
    out.requiresGrad = true;
    out._prev = prev;
    out._backward = backward;
  }
  return out;
}

// Elementwise binary op with scalar broadcasting on either side.
function broadcastBinary(
  a: Tensor,
  b: Tensor,
  f: (x: number, y: number) => number,
  // local partial derivatives dOut/dA and dOut/dB given (x, y, out)
  da: (x: number, y: number, o: number) => number,
  db: (x: number, y: number, o: number) => number,
): Tensor {
  let shape: Shape;
  if (shapeEqual(a.shape, b.shape)) shape = a.shape.slice();
  else if (a.isScalar) shape = b.shape.slice();
  else if (b.isScalar) shape = a.shape.slice();
  else {
    throw new Error(
      `shape mismatch: cannot combine [${a.shape}] with [${b.shape}] ` +
        "(only equal shapes or scalar broadcasting are supported)",
    );
  }
  const n = numel(shape);
  const out = new Float64Array(n);
  const av = (i: number) => (a.isScalar ? a.data[0] : a.data[i]);
  const bv = (i: number) => (b.isScalar ? b.data[0] : b.data[i]);
  for (let i = 0; i < n; i++) out[i] = f(av(i), bv(i));
  const res = new Tensor(out, shape);
  return tracked(res, [a, b], () => {
    const g = res.grad!;
    if (a.requiresGrad) {
      const ga = a.ensureGrad();
      for (let i = 0; i < n; i++) {
        const contrib = da(av(i), bv(i), out[i]) * g[i];
        if (a.isScalar) ga[0] += contrib;
        else ga[i] += contrib;
      }
    }
    if (b.requiresGrad) {
      const gb = b.ensureGrad();
      for (let i = 0; i < n; i++) {
        const contrib = db(av(i), bv(i), out[i]) * g[i];
        if (b.isScalar) gb[0] += contrib;
        else gb[i] += contrib;
      }
    }
  });
}

function unary(
  a: Tensor,
  f: (x: number) => number,
  df: (x: number, o: number) => number,
): Tensor {
  const n = a.data.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = f(a.data[i]);
  const res = new Tensor(out, a.shape.slice());
  return tracked(res, [a], () => {
    if (!a.requiresGrad) return;
    const ga = a.ensureGrad();
    const g = res.grad!;
    for (let i = 0; i < n; i++) ga[i] += df(a.data[i], out[i]) * g[i];
  });
}

export const add = (a: Tensor, b: Tensor) =>
  broadcastBinary(a, b, (x, y) => x + y, () => 1, () => 1);

export const sub = (a: Tensor, b: Tensor) =>
  broadcastBinary(a, b, (x, y) => x - y, () => 1, () => -1);

export const mul = (a: Tensor, b: Tensor) =>
  broadcastBinary(a, b, (x, y) => x * y, (_x, y) => y, (x) => x);

export const div = (a: Tensor, b: Tensor) =>
  broadcastBinary(
    a,
    b,
    (x, y) => x / y,
    (_x, y) => 1 / y,
    (x, y) => -x / (y * y),
  );

export const mod = (a: Tensor, b: Tensor) =>
  broadcastBinary(
    a,
    b,
    (x, y) => x - Math.floor(x / y) * y,
    () => 1,
    (x, y) => -Math.floor(x / y),
  );

export const powScalar = (a: Tensor, p: number) =>
  unary(a, (x) => Math.pow(x, p), (x) => p * Math.pow(x, p - 1));

export const neg = (a: Tensor) => unary(a, (x) => -x, () => -1);
export const relu = (a: Tensor) =>
  unary(a, (x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0));
export const exp = (a: Tensor) => unary(a, Math.exp, (_x, o) => o);
export const log = (a: Tensor) => unary(a, Math.log, (x) => 1 / x);
export const sin = (a: Tensor) => unary(a, Math.sin, (x) => Math.cos(x));
export const cos = (a: Tensor) => unary(a, Math.cos, (x) => -Math.sin(x));
export const sqrt = (a: Tensor) =>
  unary(a, Math.sqrt, (_x, o) => 0.5 / o);
export const tanh = (a: Tensor) =>
  unary(a, Math.tanh, (_x, o) => 1 - o * o);
export const sigmoid = (a: Tensor) =>
  unary(a, (x) => 1 / (1 + Math.exp(-x)), (_x, o) => o * (1 - o));

function reduceAll(a: Tensor, mean: boolean): Tensor {
  const n = a.data.length;
  let s = 0;
  for (let i = 0; i < n; i++) s += a.data[i];
  const scale = mean ? 1 / n : 1;
  const res = Tensor.scalar(s * scale);
  return tracked(res, [a], () => {
    if (!a.requiresGrad) return;
    const ga = a.ensureGrad();
    const g = res.grad![0] * scale;
    for (let i = 0; i < n; i++) ga[i] += g;
  });
}

export const sum = (a: Tensor) => reduceAll(a, false);
export const mean = (a: Tensor) => reduceAll(a, true);

// 2-D matrix multiply of raw buffers.
function mm2d(A: Float64Array, m: number, k: number, B: Float64Array, n: number): Float64Array {
  const C = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aip = A[i * k + p];
      if (aip === 0) continue;
      const bRow = p * n;
      const cRow = i * n;
      for (let j = 0; j < n; j++) C[cRow + j] += aip * B[bRow + j];
    }
  }
  return C;
}

function transpose2d(A: Float64Array, rows: number, cols: number): Float64Array {
  const T = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) T[j * rows + i] = A[i * cols + j];
  return T;
}

// Matrix / vector multiply supporting 1-D and 2-D operands (the `@` operator).
export function matmul(a: Tensor, b: Tensor): Tensor {
  const a2 = a.shape.length === 1 ? [1, a.shape[0]] : a.shape;
  const b2 = b.shape.length === 1 ? [b.shape[0], 1] : b.shape;
  if (a2.length !== 2 || b2.length !== 2) {
    throw new Error("@ (matmul) requires 1-D or 2-D operands");
  }
  const [m, k] = a2;
  const [k2, n] = b2;
  if (k !== k2) {
    throw new Error(`shape mismatch in @: [${a.shape}] @ [${b.shape}] (inner ${k} != ${k2})`);
  }
  const outData = mm2d(a.data, m, k, b.data, n);
  let outShape: Shape;
  if (a.shape.length === 1 && b.shape.length === 1) outShape = [];
  else if (a.shape.length === 1) outShape = [n];
  else if (b.shape.length === 1) outShape = [m];
  else outShape = [m, n];
  const res = new Tensor(outData, outShape);
  return tracked(res, [a, b], () => {
    const G = res.grad!; // flat length m*n
    if (a.requiresGrad) {
      // dA = G @ B^T  -> [m, k]
      const Bt = transpose2d(b.data, k, n);
      const dA = mm2d(G, m, n, Bt, k);
      const ga = a.ensureGrad();
      for (let i = 0; i < dA.length; i++) ga[i] += dA[i];
    }
    if (b.requiresGrad) {
      // dB = A^T @ G  -> [k, n]
      const At = transpose2d(a.data, m, k);
      const dB = mm2d(At, k, m, G, n);
      const gb = b.ensureGrad();
      for (let i = 0; i < dB.length; i++) gb[i] += dB[i];
    }
  });
}
