//! Bit-pattern dump of every in-scope op, for differential comparison against
//! the Go engine.
//!
//! Values are printed as raw IEEE-754 bit patterns, not decimals, because the
//! bar being checked is bit equality and a decimal formatter would hide the
//! last two bits that are exactly what this is looking for. There is a matching
//! Go program; the two outputs are compared line for line.

use raster_tensor::{Id, Tape};

fn gen(n: usize, off: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (i % 7) as f64 * 0.5 - 1.25 + i as f64 * 0.01 + off)
        .collect()
}

/// Strictly positive input, for log and sqrt.
fn genp(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.25 + (i % 5) as f64 * 0.5 + i as f64 * 0.01)
        .collect()
}

fn emit(label: &str, t: &Tape, id: Id) {
    print!("{label} shape={:?} data=", t.shape(id));
    for v in t.data(id) {
        print!("{:016x},", v.to_bits());
    }
    println!();
}

fn emit_grad(label: &str, t: &Tape, id: Id) {
    print!("{label} grad=");
    for v in t.grad(id) {
        print!("{:016x},", v.to_bits());
    }
    println!();
}

fn main() {
    // Elementwise, over every broadcast layout the implementation has a path for.
    let layouts: [(&str, Vec<usize>, Vec<usize>); 5] = [
        ("equal", vec![3, 4], vec![3, 4]),
        ("row", vec![3, 4], vec![4]),
        ("scalar-b", vec![3, 4], vec![]),
        ("scalar-a", vec![], vec![3, 4]),
        ("general", vec![2, 1, 3], vec![4, 1]),
    ];
    for (name, sa, sb) in layouts {
        let na: usize = sa.iter().product();
        let nb: usize = sb.iter().product();
        for (opname, op) in [("add", 0usize), ("sub", 1), ("mul", 2), ("div", 3)] {
            let mut t = Tape::new();
            let a = t.leaf(gen(na, 0.0), &sa);
            let b = t.leaf(genp(nb), &sb);
            let c = match op {
                0 => t.add(a, b).unwrap(),
                1 => t.sub(a, b).unwrap(),
                2 => t.mul(a, b).unwrap(),
                _ => t.div(a, b).unwrap(),
            };
            // Weight the output so the backward pass carries a different
            // gradient per element and a transposed scatter would show.
            let cn = t.data(c).len();
            let cs = t.shape(c).to_vec();
            let w = t.constant(gen(cn, 0.3), &cs);
            let p = t.mul(c, w).unwrap();
            let s = t.sum(p);
            t.backward(s).unwrap();
            emit(&format!("{opname}/{name}"), &t, c);
            emit_grad(&format!("{opname}/{name}/a"), &t, a);
            emit_grad(&format!("{opname}/{name}/b"), &t, b);
        }
    }

    // Unary.
    for (opname, positive) in [
        ("neg", false),
        ("square", false),
        ("exp", false),
        ("tanh", false),
        ("log", true),
        ("sqrt", true),
    ] {
        let mut t = Tape::new();
        let data = if positive { genp(11) } else { gen(11, 0.0) };
        let x = t.leaf(data, &[11]);
        let y = match opname {
            "neg" => t.neg(x),
            "square" => t.square(x),
            "exp" => t.exp(x),
            "tanh" => t.tanh(x),
            "log" => t.log(x),
            _ => t.sqrt(x),
        };
        let w = t.constant(gen(11, 0.7), &[11]);
        let p = t.mul(y, w).unwrap();
        let s = t.sum(p);
        t.backward(s).unwrap();
        emit(opname, &t, y);
        emit_grad(opname, &t, x);
    }

    // Reductions.
    for (name, shape) in [("vec", vec![37usize]), ("mat", vec![5, 7])] {
        let n: usize = shape.iter().product();
        let mut t = Tape::new();
        let x = t.leaf(gen(n, 0.0), &shape);
        let s = t.sum(x);
        emit(&format!("sum/{name}"), &t, s);
        let m = t.mean(x);
        emit(&format!("mean/{name}"), &t, m);
        t.backward(m).unwrap();
        emit_grad(&format!("mean/{name}"), &t, x);
    }
    for axis in [0isize, 1] {
        for mean in [false, true] {
            let mut t = Tape::new();
            let x = t.leaf(gen(35, 0.0), &[5, 7]);
            let r = if mean {
                t.mean_axis(x, axis).unwrap()
            } else {
                t.sum_axis(x, axis).unwrap()
            };
            let sq = t.square(r);
            let s = t.sum(sq);
            t.backward(s).unwrap();
            let label = format!("{}axis{axis}", if mean { "mean" } else { "sum" });
            emit(&label, &t, r);
            emit_grad(&label, &t, x);
        }
    }

    // A sum long enough to cross MIN_PARALLEL and exercise the fixed blocking.
    {
        let mut t = Tape::new();
        let x = t.constant(gen(100_003, 0.0), &[100_003]);
        let s = t.sum(x);
        emit("sum/blocked", &t, s);
    }

    // Matmul, including the rank-reducing 1-D forms.
    let cases: [(&str, Vec<usize>, Vec<usize>); 4] = [
        ("mm", vec![5, 7], vec![7, 3]),
        ("mv", vec![5, 7], vec![7]),
        ("vm", vec![5], vec![5, 3]),
        ("dot", vec![6], vec![6]),
    ];
    for (name, sa, sb) in cases {
        let na: usize = sa.iter().product();
        let nb: usize = sb.iter().product();
        let mut t = Tape::new();
        let a = t.leaf(gen(na, 0.0), &sa);
        let b = t.leaf(gen(nb, 0.4), &sb);
        let c = t.matmul(a, b).unwrap();
        let sq = t.square(c);
        let s = t.sum(sq);
        t.backward(s).unwrap();
        emit(&format!("matmul/{name}"), &t, c);
        emit_grad(&format!("matmul/{name}/a"), &t, a);
        emit_grad(&format!("matmul/{name}/b"), &t, b);
    }

    // A larger matmul, past the point where the kernel splits across threads.
    {
        let mut t = Tape::new();
        let a = t.constant(gen(128 * 96, 0.0), &[128, 96]);
        let b = t.constant(gen(96 * 64, 0.4), &[96, 64]);
        let c = t.matmul(a, b).unwrap();
        let s = t.sum(c);
        emit("matmul/threaded", &t, s);
    }

    // The mixed graph, which is the one that catches a wrong edge rather than a
    // wrong kernel.
    {
        let mut t = Tape::new();
        let x = t.leaf(gen(12, 0.0), &[3, 4]);
        let w = t.leaf(genp(8), &[4, 2]);
        let h = t.matmul(x, w).unwrap();
        let a = t.tanh(h);
        let b = t.exp(a);
        let one = t.scalar(1.0);
        let c = t.add(b, one).unwrap();
        let d = t.log(c);
        let e = t.sqrt(b);
        let f = t.div(d, e).unwrap();
        let g = t.neg(f);
        let hh = t.sub(f, g).unwrap();
        let r = t.sum_axis(hh, 1).unwrap();
        let sq = t.square(r);
        let out = t.mean(sq);
        t.backward(out).unwrap();
        emit("mixed", &t, out);
        emit_grad("mixed/x", &t, x);
        emit_grad("mixed/w", &t, w);
    }
}
