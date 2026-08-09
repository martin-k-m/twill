//! Raster's differentiable tensor engine.
//!
//! A port of the tensor core of `internal/tensor`, narrow on purpose: the
//! tensor type, the autodiff tape, elementwise arithmetic with broadcasting,
//! matmul, whole-tensor and per-axis sum and mean, four transcendentals, and
//! reverse-mode backward over any graph of them. The Go implementation is the
//! specification and stays the executable reference until this one is wider.
//!
//! Everything here is `f64` and `std` only. The empty `[dependencies]` in
//! `Cargo.toml` is the point of the project, not a detail of it.

pub mod ops;
pub mod parallel;
pub mod shape;
pub mod tape;

pub use ops::{BinKind, UnKind};
pub use shape::{broadcast_shape, numel};
pub use tape::{Id, Tape};
