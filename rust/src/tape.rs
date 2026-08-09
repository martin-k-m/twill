//! The autodiff tape as an index arena.
//!
//! The Go engine hangs the graph off `*Tensor` pointers and a per-op backward
//! closure. Ported literally that becomes `Rc<RefCell<Tensor>>`, which is
//! slower than Go and worse to read. So the graph is a `Vec<Node>` and a
//! `Tensor` is a `usize` into it.
//!
//! Two things fall out of that choice, and they are the reason it was made.
//! First, a node is always pushed after its parents, so descending index order
//! is already a topological order: the depth-first walk and the visited set in
//! `Backward` disappear. Second, nodes are contiguous, so a whole graph is
//! freed by dropping one `Vec`.
//!
//! The cost is that the arena outlives every tensor in it, so a long-running
//! program has to drop the tape between iterations rather than relying on a
//! collector. For a training loop that is one line; for the interpreter it will
//! be a scope discipline that does not exist yet.

use crate::ops::{mm, transpose2d, BinKind, UnKind};
use crate::parallel::{parallel_for, parallel_sum};
use crate::shape::{axis_spans, broadcast_shape, eff_strides, normalize_axis, numel, remove_axis};

/// A handle into the tape. Not a tensor by itself: it means nothing without the
/// tape that issued it.
pub type Id = usize;

/// How a binary op laid out its operands, recorded forward so the backward pass
/// does not have to work it out again.
enum BinMode {
    /// Shapes matched exactly.
    Equal,
    /// The right operand held a single element.
    ScalarB,
    /// The left operand held a single element.
    ScalarA,
    /// Full odometer walk. The input offsets are recorded per output element
    /// because recomputing them backwards costs more than the memory does, and
    /// this is the path that already gave up on being fast.
    General { ia: Vec<usize>, ib: Vec<usize> },
}

enum Back {
    None,
    Bin {
        kind: BinKind,
        mode: BinMode,
    },
    Un(UnKind),
    MatMul {
        m: usize,
        k: usize,
        n: usize,
    },
    ReduceAll {
        scale: f64,
    },
    ReduceAxis {
        before: usize,
        l: usize,
        after: usize,
        scale: f64,
    },
}

struct Node {
    data: Vec<f64>,
    shape: Vec<usize>,
    requires_grad: bool,
    /// Empty until a gradient actually reaches this node, matching Go's nil
    /// `Grad`. An empty gradient is the signal that the node is not on the path
    /// to the output, which is what lets the backward sweep skip it.
    grad: Vec<f64>,
    p0: Option<Id>,
    p1: Option<Id>,
    back: Back,
}

#[derive(Default)]
pub struct Tape {
    nodes: Vec<Node>,
}

fn fmt_shape(s: &[usize]) -> String {
    let mut out = String::from("[");
    for (i, d) in s.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&d.to_string());
    }
    out.push(']');
    out
}

impl Tape {
    pub fn new() -> Self {
        Tape { nodes: Vec::new() }
    }

    fn push(&mut self, data: Vec<f64>, shape: Vec<usize>) -> Id {
        self.nodes.push(Node {
            data,
            shape,
            requires_grad: false,
            grad: Vec::new(),
            p0: None,
            p1: None,
            back: Back::None,
        });
        self.nodes.len() - 1
    }

    /// A tensor that does not track gradients. Mirrors `tensor.New`.
    pub fn constant(&mut self, data: Vec<f64>, shape: &[usize]) -> Id {
        assert_eq!(data.len(), numel(shape), "data length does not match shape");
        self.push(data, shape.to_vec())
    }

    /// A rank-0 tensor.
    pub fn scalar(&mut self, x: f64) -> Id {
        self.push(vec![x], Vec::new())
    }

    /// A gradient-tracking input. Mirrors `tensor.Leaf`.
    pub fn leaf(&mut self, data: Vec<f64>, shape: &[usize]) -> Id {
        let id = self.constant(data, shape);
        self.nodes[id].requires_grad = true;
        id
    }

    pub fn data(&self, id: Id) -> &[f64] {
        &self.nodes[id].data
    }

    pub fn shape(&self, id: Id) -> &[usize] {
        &self.nodes[id].shape
    }

    pub fn grad(&self, id: Id) -> &[f64] {
        &self.nodes[id].grad
    }

    pub fn is_scalar(&self, id: Id) -> bool {
        self.nodes[id].shape.is_empty()
    }

    pub fn requires_grad(&self, id: Id) -> bool {
        self.nodes[id].requires_grad
    }

    fn ensure_grad(&mut self, id: Id) {
        if self.nodes[id].grad.is_empty() {
            self.nodes[id].grad = vec![0.0; self.nodes[id].data.len()];
        }
    }

    fn track1(&mut self, out: Id, a: Id, back: Back) -> Id {
        if self.nodes[a].requires_grad {
            self.nodes[out].requires_grad = true;
            self.nodes[out].p0 = Some(a);
            self.nodes[out].back = back;
        }
        out
    }

    fn track2(&mut self, out: Id, a: Id, b: Id, back: Back) -> Id {
        if self.nodes[a].requires_grad || self.nodes[b].requires_grad {
            self.nodes[out].requires_grad = true;
            self.nodes[out].p0 = Some(a);
            self.nodes[out].p1 = Some(b);
            self.nodes[out].back = back;
        }
        out
    }

    // ---- elementwise binary ------------------------------------------------

    pub fn add(&mut self, a: Id, b: Id) -> Result<Id, String> {
        self.binary(a, b, BinKind::Add)
    }
    pub fn sub(&mut self, a: Id, b: Id) -> Result<Id, String> {
        self.binary(a, b, BinKind::Sub)
    }
    pub fn mul(&mut self, a: Id, b: Id) -> Result<Id, String> {
        self.binary(a, b, BinKind::Mul)
    }
    pub fn div(&mut self, a: Id, b: Id) -> Result<Id, String> {
        self.binary(a, b, BinKind::Div)
    }

    fn binary(&mut self, a: Id, b: Id, kind: BinKind) -> Result<Id, String> {
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;

        // Two scalars with nothing to differentiate is what an interpreted loop
        // is made of, and the general path charges it a broadcast computation
        // and a thread decision to produce one number. Kept from the Go code
        // because the interpreter will land on it just as hard.
        if !rg && self.nodes[a].shape.is_empty() && self.nodes[b].shape.is_empty() {
            let v = kind.f(self.nodes[a].data[0], self.nodes[b].data[0]);
            return Ok(self.scalar(v));
        }

        let shape =
            broadcast_shape(&self.nodes[a].shape, &self.nodes[b].shape).ok_or_else(|| {
                format!(
                    "shape mismatch: cannot broadcast {} with {}",
                    fmt_shape(&self.nodes[a].shape),
                    fmt_shape(&self.nodes[b].shape)
                )
            })?;
        let n = numel(&shape);
        let mut out = vec![0.0; n];

        let mode = {
            let ad = &self.nodes[a].data;
            let bd = &self.nodes[b].data;
            if self.nodes[a].shape == self.nodes[b].shape {
                parallel_for(n, &mut out, |lo, chunk| {
                    for (i, o) in chunk.iter_mut().enumerate() {
                        *o = kind.f(ad[lo + i], bd[lo + i]);
                    }
                });
                BinMode::Equal
            } else if bd.len() == 1 {
                let bs = bd[0];
                parallel_for(n, &mut out, |lo, chunk| {
                    for (i, o) in chunk.iter_mut().enumerate() {
                        *o = kind.f(ad[lo + i], bs);
                    }
                });
                BinMode::ScalarB
            } else if ad.len() == 1 {
                let as_ = ad[0];
                parallel_for(n, &mut out, |lo, chunk| {
                    for (i, o) in chunk.iter_mut().enumerate() {
                        *o = kind.f(as_, bd[lo + i]);
                    }
                });
                BinMode::ScalarA
            } else {
                // Odometer walk: input offsets update incrementally so there is
                // no division per element.
                let rank = shape.len();
                let eff_a = eff_strides(&self.nodes[a].shape, &shape);
                let eff_b = eff_strides(&self.nodes[b].shape, &shape);
                let mut coord = vec![0usize; rank];
                let (mut ia, mut ib) = (0usize, 0usize);
                let (mut ia_hist, mut ib_hist) = if rg {
                    (vec![0usize; n], vec![0usize; n])
                } else {
                    (Vec::new(), Vec::new())
                };
                for (o, slot) in out.iter_mut().enumerate() {
                    if rg {
                        ia_hist[o] = ia;
                        ib_hist[o] = ib;
                    }
                    *slot = kind.f(ad[ia], bd[ib]);
                    for d in (0..rank).rev() {
                        coord[d] += 1;
                        ia += eff_a[d];
                        ib += eff_b[d];
                        if coord[d] < shape[d] {
                            break;
                        }
                        coord[d] = 0;
                        ia -= eff_a[d] * shape[d];
                        ib -= eff_b[d] * shape[d];
                    }
                }
                BinMode::General {
                    ia: ia_hist,
                    ib: ib_hist,
                }
            }
        };

        let res = self.push(out, shape);
        if !rg {
            return Ok(res);
        }
        Ok(self.track2(res, a, b, Back::Bin { kind, mode }))
    }

    // ---- elementwise unary -------------------------------------------------

    pub fn neg(&mut self, a: Id) -> Id {
        self.unary(a, UnKind::Neg)
    }
    pub fn square(&mut self, a: Id) -> Id {
        self.unary(a, UnKind::Square)
    }
    pub fn exp(&mut self, a: Id) -> Id {
        self.unary(a, UnKind::Exp)
    }
    pub fn log(&mut self, a: Id) -> Id {
        self.unary(a, UnKind::Log)
    }
    pub fn sqrt(&mut self, a: Id) -> Id {
        self.unary(a, UnKind::Sqrt)
    }
    pub fn tanh(&mut self, a: Id) -> Id {
        self.unary(a, UnKind::Tanh)
    }

    fn unary(&mut self, a: Id, kind: UnKind) -> Id {
        let n = self.nodes[a].data.len();
        let mut out = vec![0.0; n];
        {
            let ad = &self.nodes[a].data;
            parallel_for(n, &mut out, |lo, chunk| {
                for (i, o) in chunk.iter_mut().enumerate() {
                    *o = kind.f(ad[lo + i]);
                }
            });
        }
        let shape = self.nodes[a].shape.clone();
        let res = self.push(out, shape);
        self.track1(res, a, Back::Un(kind))
    }

    // ---- reductions --------------------------------------------------------

    pub fn sum(&mut self, a: Id) -> Id {
        self.reduce_all(a, false)
    }
    pub fn mean(&mut self, a: Id) -> Id {
        self.reduce_all(a, true)
    }

    fn reduce_all(&mut self, a: Id, mean: bool) -> Id {
        let n = self.nodes[a].data.len();
        let s = parallel_sum(&self.nodes[a].data);
        let scale = if mean { 1.0 / n as f64 } else { 1.0 };
        let res = self.push(vec![s * scale], Vec::new());
        self.track1(res, a, Back::ReduceAll { scale })
    }

    pub fn sum_axis(&mut self, a: Id, axis: isize) -> Result<Id, String> {
        self.reduce_axis(a, axis, false)
    }
    pub fn mean_axis(&mut self, a: Id, axis: isize) -> Result<Id, String> {
        self.reduce_axis(a, axis, true)
    }

    fn reduce_axis(&mut self, t: Id, axis: isize, mean: bool) -> Result<Id, String> {
        let axis = normalize_axis(axis, self.nodes[t].shape.len())?;
        let (before, l, after) = axis_spans(&self.nodes[t].shape, axis);
        let out_n = before * after;
        let mut out = vec![0.0; out_n];
        let scale = if mean { 1.0 / l as f64 } else { 1.0 };
        {
            let td = &self.nodes[t].data;
            for i in 0..before {
                for j in 0..after {
                    let base = i * l * after + j;
                    let mut s = 0.0;
                    // Serial and in axis order. The whole-tensor reduction gets
                    // the blocked sum; this one does not, because matching the
                    // Go accumulation order matters more than the speed of a
                    // reduction whose runs are usually short.
                    for k in 0..l {
                        s += td[base + k * after];
                    }
                    out[i * after + j] = s * scale;
                }
            }
        }
        let shape = remove_axis(&self.nodes[t].shape, axis);
        let res = self.push(out, shape);
        Ok(self.track1(
            res,
            t,
            Back::ReduceAxis {
                before,
                l,
                after,
                scale,
            },
        ))
    }

    // ---- matmul ------------------------------------------------------------

    /// The `@` operator: 1-D or 2-D operands only.
    pub fn matmul(&mut self, a: Id, b: Id) -> Result<Id, String> {
        let a_shape = self.nodes[a].shape.clone();
        let b_shape = self.nodes[b].shape.clone();
        let a2: Vec<usize> = if a_shape.len() == 1 {
            vec![1, a_shape[0]]
        } else {
            a_shape.clone()
        };
        let b2: Vec<usize> = if b_shape.len() == 1 {
            vec![b_shape[0], 1]
        } else {
            b_shape.clone()
        };
        if a2.len() != 2 || b2.len() != 2 {
            return Err("@ (matmul) requires 1-D or 2-D operands".to_string());
        }
        let (m, k) = (a2[0], a2[1]);
        let (k2, n) = (b2[0], b2[1]);
        if k != k2 {
            return Err(format!(
                "shape mismatch in @: {} @ {} (inner {} != {})",
                fmt_shape(&a_shape),
                fmt_shape(&b_shape),
                k,
                k2
            ));
        }
        let out = mm(&self.nodes[a].data, m, k, &self.nodes[b].data, n);
        let out_shape: Vec<usize> = match (a_shape.len(), b_shape.len()) {
            (1, 1) => vec![],
            (1, _) => vec![n],
            (_, 1) => vec![m],
            _ => vec![m, n],
        };
        let res = self.push(out, out_shape);
        Ok(self.track2(res, a, b, Back::MatMul { m, k, n }))
    }

    // ---- backward ----------------------------------------------------------

    /// Reverse-mode backpropagation from a scalar output.
    pub fn backward(&mut self, id: Id) -> Result<(), String> {
        if !self.nodes[id].shape.is_empty() {
            return Err("backward may only be called on a scalar output".to_string());
        }
        self.ensure_grad(id);
        self.nodes[id].grad[0] = 1.0;
        // Descending index order is a topological order, because a node cannot
        // be pushed before its parents. Nodes that no gradient reached still
        // have an empty gradient and are skipped; they would contribute zero.
        for i in (0..=id).rev() {
            if self.nodes[i].grad.is_empty() {
                continue;
            }
            self.backward_node(i);
        }
        Ok(())
    }

    /// Moves the node's own buffers out of the arena for the duration of its
    /// backward step, so that reading this node and writing its parents are two
    /// disjoint borrows. Moving three `Vec` headers is the price of not
    /// threading a lifetime or a `RefCell` through the whole tape.
    fn backward_node(&mut self, i: Id) {
        let back = std::mem::replace(&mut self.nodes[i].back, Back::None);
        let out = std::mem::take(&mut self.nodes[i].data);
        let g = std::mem::take(&mut self.nodes[i].grad);
        let (p0, p1) = (self.nodes[i].p0, self.nodes[i].p1);
        match &back {
            Back::None => {}
            Back::Bin { kind, mode } => {
                self.bin_backward(*kind, mode, p0.unwrap(), p1.unwrap(), &out, &g)
            }
            Back::Un(kind) => self.un_backward(*kind, p0.unwrap(), &out, &g),
            Back::MatMul { m, k, n } => {
                self.matmul_backward(p0.unwrap(), p1.unwrap(), *m, *k, *n, &g)
            }
            Back::ReduceAll { scale } => {
                let a = p0.unwrap();
                if self.nodes[a].requires_grad {
                    self.ensure_grad(a);
                    let gv = g[0] * scale;
                    let n = self.nodes[a].grad.len();
                    let mut ga = std::mem::take(&mut self.nodes[a].grad);
                    parallel_for(n, &mut ga, |_, chunk| {
                        for slot in chunk.iter_mut() {
                            *slot += gv;
                        }
                    });
                    self.nodes[a].grad = ga;
                }
            }
            Back::ReduceAxis {
                before,
                l,
                after,
                scale,
            } => {
                let t = p0.unwrap();
                if self.nodes[t].requires_grad {
                    self.ensure_grad(t);
                    let gt = &mut self.nodes[t].grad;
                    for i2 in 0..*before {
                        for j in 0..*after {
                            let gv = g[i2 * after + j] * scale;
                            let base = i2 * l * after + j;
                            for k in 0..*l {
                                gt[base + k * after] += gv;
                            }
                        }
                    }
                }
            }
        }
        self.nodes[i].data = out;
        self.nodes[i].grad = g;
        self.nodes[i].back = back;
    }

    fn un_backward(&mut self, kind: UnKind, a: Id, out: &[f64], g: &[f64]) {
        if !self.nodes[a].requires_grad {
            return;
        }
        self.ensure_grad(a);
        let ad = std::mem::take(&mut self.nodes[a].data);
        let mut ga = std::mem::take(&mut self.nodes[a].grad);
        let n = ad.len();
        parallel_for(n, &mut ga, |lo, chunk| {
            for (i, slot) in chunk.iter_mut().enumerate() {
                let k = lo + i;
                *slot += kind.df(ad[k], out[k]) * g[k];
            }
        });
        self.nodes[a].data = ad;
        self.nodes[a].grad = ga;
    }

    fn bin_backward(
        &mut self,
        kind: BinKind,
        mode: &BinMode,
        a: Id,
        b: Id,
        out: &[f64],
        g: &[f64],
    ) {
        let a_req = self.nodes[a].requires_grad;
        let b_req = self.nodes[b].requires_grad;
        if a_req {
            self.ensure_grad(a);
        }
        if b_req {
            self.ensure_grad(b);
        }
        let ad = std::mem::take(&mut self.nodes[a].data);
        // When both operands are the same node (`x * x`) there is one buffer,
        // and the two accumulation passes run one after the other, which is what
        // Go's aliased `ga` and `gb` do too.
        let bd = if a == b {
            Vec::new()
        } else {
            std::mem::take(&mut self.nodes[b].data)
        };
        let bref: &[f64] = if a == b { &ad } else { &bd };
        let n = out.len();

        if a_req {
            let mut ga = std::mem::take(&mut self.nodes[a].grad);
            match mode {
                BinMode::Equal => parallel_for(n, &mut ga, |lo, chunk| {
                    for (i, slot) in chunk.iter_mut().enumerate() {
                        let k = lo + i;
                        *slot += kind.da(ad[k], bref[k], out[k]) * g[k];
                    }
                }),
                BinMode::ScalarB => {
                    let bs = bref[0];
                    for i in 0..n {
                        ga[i] += kind.da(ad[i], bs, out[i]) * g[i];
                    }
                }
                BinMode::ScalarA => {
                    let as_ = ad[0];
                    for i in 0..n {
                        ga[0] += kind.da(as_, bref[i], out[i]) * g[i];
                    }
                }
                BinMode::General { ia, ib } => {
                    for o in 0..n {
                        ga[ia[o]] += kind.da(ad[ia[o]], bref[ib[o]], out[o]) * g[o];
                    }
                }
            }
            self.nodes[a].grad = ga;
        }
        if b_req {
            let mut gb = std::mem::take(&mut self.nodes[b].grad);
            match mode {
                BinMode::Equal => parallel_for(n, &mut gb, |lo, chunk| {
                    for (i, slot) in chunk.iter_mut().enumerate() {
                        let k = lo + i;
                        *slot += kind.db(ad[k], bref[k], out[k]) * g[k];
                    }
                }),
                BinMode::ScalarB => {
                    let bs = bref[0];
                    for i in 0..n {
                        gb[0] += kind.db(ad[i], bs, out[i]) * g[i];
                    }
                }
                BinMode::ScalarA => {
                    let as_ = ad[0];
                    for i in 0..n {
                        gb[i] += kind.db(as_, bref[i], out[i]) * g[i];
                    }
                }
                BinMode::General { ia, ib } => {
                    for o in 0..n {
                        gb[ib[o]] += kind.db(ad[ia[o]], bref[ib[o]], out[o]) * g[o];
                    }
                }
            }
            self.nodes[b].grad = gb;
        }

        self.nodes[a].data = ad;
        if a != b {
            self.nodes[b].data = bd;
        }
    }

    fn matmul_backward(&mut self, a: Id, b: Id, m: usize, k: usize, n: usize, g: &[f64]) {
        let a_req = self.nodes[a].requires_grad;
        let b_req = self.nodes[b].requires_grad;
        if a_req {
            self.ensure_grad(a);
        }
        if b_req {
            self.ensure_grad(b);
        }
        let ad = std::mem::take(&mut self.nodes[a].data);
        let bd = if a == b {
            Vec::new()
        } else {
            std::mem::take(&mut self.nodes[b].data)
        };
        let bref: &[f64] = if a == b { &ad } else { &bd };

        if a_req {
            let bt = transpose2d(bref, k, n);
            let da = mm(g, m, n, &bt, k);
            let ga = &mut self.nodes[a].grad;
            for (slot, v) in ga.iter_mut().zip(da.iter()) {
                *slot += v;
            }
        }
        if b_req {
            let at = transpose2d(&ad, m, k);
            let db = mm(&at, k, m, g, n);
            let gb = &mut self.nodes[b].grad;
            for (slot, v) in gb.iter_mut().zip(db.iter()) {
                *slot += v;
            }
        }

        self.nodes[a].data = ad;
        if a != b {
            self.nodes[b].data = bd;
        }
    }
}
