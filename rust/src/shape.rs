//! Shape arithmetic, ported from `tensor.go` and `ops.go`.

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major strides.
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        s[i] = acc;
        acc *= shape[i];
    }
    s
}

/// NumPy-style broadcast of two shapes, aligned from the right. Dimensions must
/// be equal or one of them must be 1. Returns `None` when they do not combine.
///
/// Ported dimension for dimension from `BroadcastShape`, including the detail
/// that a missing leading dimension is treated as 1 rather than as an error.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let (ra, rb) = (a.len(), b.len());
    let r = ra.max(rb);
    let mut out = vec![0usize; r];
    for i in 0..r {
        let da = if i < ra { a[ra - 1 - i] } else { 1 };
        let db = if i < rb { b[rb - 1 - i] } else { 1 };
        out[r - 1 - i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return None;
        };
    }
    Some(out)
}

/// For each dimension of `out_shape`, the stride to advance in `in_shape`'s flat
/// data. Zero where `in_shape` broadcasts (a size-1 or absent dimension).
pub fn eff_strides(in_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let (r_in, r_out) = (in_shape.len(), out_shape.len());
    let real = strides(in_shape);
    let mut eff = vec![0usize; r_out];
    for i in 0..r_out {
        let j = r_out - 1 - i;
        if i < r_in && in_shape[r_in - 1 - i] != 1 {
            eff[j] = real[r_in - 1 - i];
        }
    }
    eff
}

/// Splits a shape around `axis` into the product of the dimensions before it,
/// the axis length, and the product of those after.
pub fn axis_spans(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    let before: usize = shape[..axis].iter().product();
    let after: usize = shape[axis + 1..].iter().product();
    (before, shape[axis], after)
}

pub fn remove_axis(shape: &[usize], axis: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(shape.len() - 1);
    out.extend_from_slice(&shape[..axis]);
    out.extend_from_slice(&shape[axis + 1..]);
    out
}

/// Resolves a possibly negative axis against a rank, matching `normalizeAxis`
/// including its error text, because that text is part of the product.
pub fn normalize_axis(axis: isize, rank: usize) -> Result<usize, String> {
    let a = if axis < 0 { axis + rank as isize } else { axis };
    if a < 0 || a >= rank as isize {
        return Err(format!("axis {axis} out of range for rank {rank}"));
    }
    Ok(a as usize)
}
