//! Matmul throughput, comparable with the Go engine's measured figure.
//!
//! Run with `cargo run --release --bin bench`. The kernel here is the same
//! naive i-p-j loop the Go code uses, so the number this prints is the cost of
//! the language and the code generator, not of a better algorithm. Tiling and
//! explicit vectorisation are a later stage and are where the plan's 18 to 27
//! GFLOP/s is supposed to come from.

use raster_tensor::ops::mm;
use std::time::Instant;

fn main() {
    for n in [64usize, 128, 256, 512] {
        let a: Vec<f64> = (0..n * n).map(|i| (i % 7) as f64 * 0.5).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (i % 7) as f64 * 0.5).collect();

        // One untimed pass so the measurement is not paying for first-touch
        // page faults on the output buffer.
        let warm = mm(&a, n, n, &b, n);
        std::hint::black_box(&warm);

        // A wall-clock budget rather than a fixed iteration count. At 512 a
        // flop-derived count lands on three iterations, and three samples of a
        // 50 ms operation on a laptop is noise, not a measurement.
        let budget = std::time::Duration::from_secs(3);
        let start = Instant::now();
        let mut iters = 0usize;
        while start.elapsed() < budget || iters < 5 {
            let c = mm(&a, n, n, &b, n);
            std::hint::black_box(&c);
            iters += 1;
        }
        let elapsed = start.elapsed().as_secs_f64() / iters as f64;
        let gflops = 2.0 * (n * n * n) as f64 / elapsed / 1e9;
        println!(
            "matmul {n}x{n}: {:.3} ms/op, {gflops:.2} GFLOP/s ({iters} iterations)",
            elapsed * 1e3
        );
    }
}
