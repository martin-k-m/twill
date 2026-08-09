//! Forward-value and shape semantics that must match `internal/tensor` exactly.

use raster_tensor::{broadcast_shape, Tape};

#[test]
fn broadcasting_aligns_shapes_from_the_right() {
    assert_eq!(broadcast_shape(&[2, 3], &[3]), Some(vec![2, 3]));
    assert_eq!(broadcast_shape(&[2, 1], &[3]), Some(vec![2, 3]));
    assert_eq!(broadcast_shape(&[2, 1, 3], &[4, 1]), Some(vec![2, 4, 3]));
    assert_eq!(broadcast_shape(&[], &[2, 2]), Some(vec![2, 2]));
    assert_eq!(broadcast_shape(&[5], &[1]), Some(vec![5]));
}

#[test]
fn broadcasting_refuses_dimensions_that_are_neither_equal_nor_one() {
    assert_eq!(broadcast_shape(&[2, 3], &[4]), None);
    assert_eq!(broadcast_shape(&[3, 2], &[2, 3]), None);
}

#[test]
fn a_general_broadcast_produces_the_same_values_as_the_odometer_implies() {
    let mut t = Tape::new();
    let a = t.constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    let b = t.constant(vec![10.0, 20.0], &[2, 1]);
    let c = t.add(a, b).unwrap();
    assert_eq!(t.shape(c), &[2, 2, 3]);
    assert_eq!(
        t.data(c),
        &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 14.0, 15.0, 16.0, 24.0, 25.0, 26.0]
    );
}

#[test]
fn matmul_shapes_follow_the_rank_of_the_operands() {
    let mut t = Tape::new();
    let m = t.constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let v = t.constant(vec![1.0, 1.0, 1.0], &[3]);
    let mv = t.matmul(m, v).unwrap();
    assert_eq!(t.shape(mv), &[2]);
    assert_eq!(t.data(mv), &[6.0, 15.0]);

    let r = t.constant(vec![1.0, 2.0], &[2]);
    let rm = t.matmul(r, m).unwrap();
    assert_eq!(t.shape(rm), &[3]);
    assert_eq!(t.data(rm), &[9.0, 12.0, 15.0]);

    let dot = t.matmul(v, v).unwrap();
    assert!(t.shape(dot).is_empty());
    assert_eq!(t.data(dot), &[3.0]);
}

#[test]
fn matmul_reports_the_inner_dimension_it_could_not_match() {
    let mut t = Tape::new();
    let a = t.constant(vec![1.0, 2.0], &[1, 2]);
    let b = t.constant(vec![1.0, 2.0, 3.0], &[3, 1]);
    assert_eq!(
        t.matmul(a, b).unwrap_err(),
        "shape mismatch in @: [1 2] @ [3 1] (inner 2 != 3)"
    );
}

#[test]
fn an_axis_reduction_resolves_a_negative_axis_from_the_end() {
    let mut t = Tape::new();
    let x = t.constant(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let r = t.mean_axis(x, -1).unwrap();
    assert_eq!(t.data(r), &[1.5, 3.5]);
    assert_eq!(t.shape(r), &[2]);
}

#[test]
fn an_axis_reduction_rejects_an_axis_outside_the_rank() {
    let mut t = Tape::new();
    let x = t.constant(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(
        t.sum_axis(x, 5).unwrap_err(),
        "axis 5 out of range for rank 2"
    );
    assert!(t.mean_axis(x, -5).is_err());
}

#[test]
fn backward_refuses_a_non_scalar_output() {
    let mut t = Tape::new();
    let x = t.leaf(vec![1.0, 2.0], &[2]);
    let y = t.square(x);
    assert_eq!(
        t.backward(y).unwrap_err(),
        "backward may only be called on a scalar output"
    );
}

#[test]
fn a_blocked_sum_gives_the_same_answer_whatever_the_core_count() {
    // The 4096-element blocking is fixed rather than derived from the machine,
    // so this value is a constant of the implementation, not of the host.
    let n = 100_000usize;
    let data: Vec<f64> = (0..n).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let mut expected = 0.0;
    let nblocks = n.div_ceil(4096);
    let mut partials = Vec::new();
    for blk in 0..nblocks {
        let start = blk * 4096;
        let end = (start + 4096).min(n);
        let mut s = 0.0;
        for &x in &data[start..end] {
            s += x;
        }
        partials.push(s);
    }
    for p in partials {
        expected += p;
    }
    let mut t = Tape::new();
    let x = t.constant(data, &[n]);
    let s = t.sum(x);
    assert_eq!(t.data(s)[0].to_bits(), expected.to_bits());
}

#[test]
fn ops_without_a_grad_tracking_input_build_no_tape_edges() {
    let mut t = Tape::new();
    let a = t.constant(vec![2.0], &[]);
    let b = t.constant(vec![3.0], &[]);
    let c = t.mul(a, b).unwrap();
    assert!(!t.requires_grad(c));
    assert_eq!(t.data(c), &[6.0]);
}
