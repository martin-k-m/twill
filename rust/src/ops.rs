//! Op kernels and their local derivatives.
//!
//! The Go original passes closures (`f`, `da`, `db`) into a generic
//! `broadcastBinary`. Closures cannot live in an index arena without dragging a
//! lifetime through every node, so the kind is an enum and the derivative is a
//! match. The cost is a branch per element in the cold paths and a devirtualised
//! loop in the hot ones, which is the direction worth trading in.

use crate::parallel::{run_chunks, workers_for};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinKind {
    #[inline]
    pub fn f(self, x: f64, y: f64) -> f64 {
        match self {
            BinKind::Add => x + y,
            BinKind::Sub => x - y,
            BinKind::Mul => x * y,
            BinKind::Div => x / y,
        }
    }

    /// d out / d x, given the forward output `o` as well, because several ops
    /// get their derivative more cheaply from the result than from the inputs.
    #[inline]
    pub fn da(self, _x: f64, y: f64, _o: f64) -> f64 {
        match self {
            BinKind::Add | BinKind::Sub => 1.0,
            BinKind::Mul => y,
            BinKind::Div => 1.0 / y,
        }
    }

    #[inline]
    pub fn db(self, x: f64, y: f64, _o: f64) -> f64 {
        match self {
            BinKind::Add => 1.0,
            BinKind::Sub => -1.0,
            BinKind::Mul => x,
            BinKind::Div => -x / (y * y),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnKind {
    Neg,
    Square,
    Exp,
    Log,
    Sqrt,
    Tanh,
}

impl UnKind {
    #[inline]
    pub fn f(self, x: f64) -> f64 {
        match self {
            UnKind::Neg => -x,
            UnKind::Square => x * x,
            UnKind::Exp => x.exp(),
            UnKind::Log => x.ln(),
            UnKind::Sqrt => x.sqrt(),
            UnKind::Tanh => x.tanh(),
        }
    }

    #[inline]
    pub fn df(self, x: f64, o: f64) -> f64 {
        match self {
            UnKind::Neg => -1.0,
            UnKind::Square => 2.0 * x,
            // exp and tanh and sqrt read the output rather than recomputing the
            // forward value, which is what the Go version does and is therefore
            // also what the last bit of the answer depends on.
            UnKind::Exp => o,
            UnKind::Log => 1.0 / x,
            UnKind::Sqrt => 0.5 / o,
            UnKind::Tanh => 1.0 - o * o,
        }
    }
}

/// Row-major matrix product, `a` is m x k and `b` is k x n.
///
/// The zero skip is not an optimisation detail that can be dropped: it changes
/// the answer when a row holds a zero and `b` holds an infinity or a NaN, and
/// the Go implementation has it, so agreement requires it here.
pub fn mm(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    run_chunks(m, workers_for(m * k * n), n, &mut c, |lo, rows| {
        for (ii, row) in rows.chunks_mut(n).enumerate() {
            let i = lo + ii;
            for p in 0..k {
                let aip = a[i * k + p];
                if aip == 0.0 {
                    continue;
                }
                let brow = &b[p * n..p * n + n];
                for j in 0..n {
                    row[j] += aip * brow[j];
                }
            }
        }
    });
    c
}

pub fn transpose2d(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}
