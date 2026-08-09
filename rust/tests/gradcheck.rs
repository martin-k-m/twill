//! Finite-difference gradient checks, ported from
//! `internal/tensor/gradcheck_test.go`.
//!
//! Every case in that file that falls inside this port's op set is here with
//! the same data, the same eps and the same bar. The remaining cases in the Go
//! file cover ops this stage deliberately does not implement (conv, pool,
//! einsum, sort, topk, gather, scans, split, concat, where, clip, prod) and
//! come back when those do.
//!
//! Finite differences are the check that does not care which implementation is
//! right, only that the analytic gradient matches the numeric one. Two
//! implementations agreeing on the same wrong derivative is the failure mode it
//! exists to catch.

use raster_tensor::{Id, Tape};

const EPS: f64 = 1e-6;
const BAR: f64 = 1e-4;

fn grad_check<F>(name: &str, data: &[f64], shape: &[usize], build: F)
where
    F: Fn(&mut Tape, Id) -> Id,
{
    let mut tape = Tape::new();
    let leaf = tape.leaf(data.to_vec(), shape);
    let out = build(&mut tape, leaf);
    assert!(
        tape.is_scalar(out),
        "{name}: build must return a scalar, got shape {:?}",
        tape.shape(out)
    );
    tape.backward(out)
        .unwrap_or_else(|e| panic!("{name}: backward: {e}"));
    let analytic = tape.grad(leaf).to_vec();

    for i in 0..data.len() {
        let eval = |delta: f64| -> f64 {
            let mut perturbed = data.to_vec();
            perturbed[i] += delta;
            let mut t = Tape::new();
            let x = t.constant(perturbed, shape);
            let o = build(&mut t, x);
            t.data(o)[0]
        };
        let numeric = (eval(EPS) - eval(-EPS)) / (2.0 * EPS);
        assert!(
            (numeric - analytic[i]).abs() <= BAR,
            "{name}: grad[{i}] analytic={} numeric={numeric}",
            analytic[i]
        );
    }
}

// ---- cases carried over verbatim from gradcheck_test.go --------------------

#[test]
fn a_row_broadcast_through_multiply_reaches_every_element() {
    // f(x) = sum(x * row) where row broadcasts over the rows of x.
    grad_check("broadcast-mul", &[1.0, 2.0, 3.0, 4.0], &[2, 2], |t, x| {
        let row = t.constant(vec![2.0, 3.0], &[2]);
        let p = t.mul(x, row).unwrap();
        t.sum(p)
    });
}

#[test]
fn a_column_vector_broadcasts_across_a_matrix_in_addition() {
    grad_check(
        "broadcast-add-col",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        |t, x| {
            let col = t.constant(vec![5.0, 7.0], &[2, 1]);
            let s = t.add(x, col).unwrap();
            t.sum(s)
        },
    );
}

#[test]
fn squaring_then_averaging_differentiates_correctly() {
    grad_check("square-mean", &[-1.0, 0.5, 2.0, -3.0], &[4], |t, x| {
        let sq = t.square(x);
        t.mean(sq)
    });
}

#[test]
fn summing_along_an_axis_scatters_the_gradient_back_along_it() {
    grad_check(
        "sum-axis",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        |t, x| {
            let r = t.sum_axis(x, 1).unwrap();
            let sq = t.square(r);
            t.sum(sq)
        },
    );
}

#[test]
fn averaging_along_an_axis_divides_the_scattered_gradient() {
    grad_check(
        "mean-axis",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        |t, x| {
            let r = t.mean_axis(x, 0).unwrap();
            let sq = t.square(r);
            t.sum(sq)
        },
    );
}

#[test]
fn matmul_routes_the_gradient_through_the_transpose() {
    grad_check(
        "matmul",
        &[1.0, 0.0, -1.0, 2.0, 1.0, 1.0],
        &[2, 3],
        |t, x| {
            let b = t.constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
            let m = t.matmul(x, b).unwrap();
            let sq = t.square(m);
            t.sum(sq)
        },
    );
}

// ---- cases the Go file does not have, for ops this stage adds --------------

#[test]
fn subtraction_negates_the_gradient_of_its_right_operand() {
    grad_check("sub", &[1.0, -2.0, 0.5, 3.0], &[4], |t, x| {
        let c = t.constant(vec![0.5, 1.5, -2.0, 4.0], &[4]);
        let d = t.sub(c, x).unwrap();
        let sq = t.square(d);
        t.sum(sq)
    });
}

#[test]
fn division_differentiates_in_both_numerator_and_denominator() {
    grad_check("div-num", &[1.0, -2.0, 0.5, 3.0], &[4], |t, x| {
        let c = t.constant(vec![0.5, 1.5, -2.0, 4.0], &[4]);
        let d = t.div(x, c).unwrap();
        t.sum(d)
    });
    grad_check("div-den", &[1.0, -2.0, 0.5, 3.0], &[4], |t, x| {
        let c = t.constant(vec![0.5, 1.5, -2.0, 4.0], &[4]);
        let d = t.div(c, x).unwrap();
        t.sum(d)
    });
}

#[test]
fn negation_flips_the_sign_of_the_gradient() {
    grad_check("neg", &[1.0, -2.0, 0.5, 3.0], &[4], |t, x| {
        let n = t.neg(x);
        let sq = t.square(n);
        t.sum(sq)
    });
}

#[test]
fn exp_log_sqrt_and_tanh_each_match_their_numeric_derivative() {
    grad_check("exp", &[0.3, -1.2, 0.0, 2.0], &[4], |t, x| {
        let e = t.exp(x);
        t.sum(e)
    });
    grad_check("log", &[0.3, 1.2, 2.5, 4.0], &[4], |t, x| {
        let l = t.log(x);
        t.sum(l)
    });
    grad_check("sqrt", &[0.3, 1.2, 2.5, 4.0], &[4], |t, x| {
        let s = t.sqrt(x);
        t.sum(s)
    });
    grad_check("tanh", &[0.3, -1.2, 0.0, 2.0], &[4], |t, x| {
        let h = t.tanh(x);
        t.sum(h)
    });
}

#[test]
fn a_general_broadcast_with_neither_side_scalar_still_differentiates() {
    // [2,1,3] against [4,1]: both operands stretch, so the odometer path runs
    // and the recorded offsets are the only route back to the inputs.
    grad_check(
        "general-broadcast",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 1, 3],
        |t, x| {
            let c = t.constant(vec![0.5, -1.0, 2.0, 3.0], &[4, 1]);
            let p = t.mul(x, c).unwrap();
            let sq = t.square(p);
            t.sum(sq)
        },
    );
}

#[test]
fn a_scalar_operand_accumulates_gradient_from_every_element() {
    grad_check("scalar-broadcast", &[2.0], &[], |t, x| {
        let m = t.constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let p = t.mul(m, x).unwrap();
        let sq = t.square(p);
        t.sum(sq)
    });
}

#[test]
fn a_node_used_twice_accumulates_both_contributions() {
    grad_check("shared-node", &[1.5, -0.5, 2.0, 0.25], &[4], |t, x| {
        let p = t.mul(x, x).unwrap();
        t.sum(p)
    });
}

#[test]
fn a_deep_mixed_graph_of_every_op_differentiates_end_to_end() {
    // One expression touching add, sub, mul, div, neg, exp, log, sqrt, tanh,
    // matmul, sum_axis and mean, so that a wrong edge anywhere in the tape
    // shows up as a wrong number here.
    grad_check(
        "mixed",
        &[0.5, 1.5, -0.75, 2.0, 0.25, 1.25],
        &[2, 3],
        |t, x| {
            let w = t.constant(vec![0.5, -1.0, 2.0, 0.25, 1.5, -0.5], &[3, 2]);
            let h = t.matmul(x, w).unwrap(); // [2,2]
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
            t.mean(sq)
        },
    );
}

#[test]
fn a_matmul_against_a_vector_differentiates_on_both_sides() {
    grad_check(
        "matvec",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        |t, x| {
            let v = t.constant(vec![0.5, -1.0, 2.0], &[3]);
            let m = t.matmul(x, v).unwrap();
            let sq = t.square(m);
            t.sum(sq)
        },
    );
    grad_check("vecmat", &[1.0, 2.0, 3.0], &[3], |t, x| {
        let m0 = t.constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let m = t.matmul(x, m0).unwrap();
        let sq = t.square(m);
        t.sum(sq)
    });
}

#[test]
fn a_tensor_large_enough_to_be_split_across_threads_still_differentiates() {
    // Above MIN_PARALLEL, so the threaded forward and threaded backward paths
    // are the ones under test rather than the serial fallbacks.
    let n = 9000usize;
    let data: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 * 0.25).collect();
    let mut tape = Tape::new();
    let leaf = tape.leaf(data.clone(), &[n]);
    let sq = tape.square(leaf);
    let out = tape.sum(sq);
    tape.backward(out).unwrap();
    for (i, &g) in tape.grad(leaf).iter().enumerate() {
        assert!((g - 2.0 * data[i]).abs() < 1e-12, "grad[{i}] = {g}");
    }
}
