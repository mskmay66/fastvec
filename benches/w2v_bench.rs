use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use fastvec::word2vec::W2V;

fn forward_benchmark(c: &mut Criterion) {
    let input = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let target = Array2::from_shape_vec((5, 1), vec![5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    let mut w2v = W2V::new(5, 0.01);
    c.bench_function("forward", |b| {
        b.iter(|| w2v.forward(input.view(), target.view()))
    });
}

fn backward_benchmark(c: &mut Criterion) {
    let input = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let target = Array2::from_shape_vec((5, 1), vec![5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    let mut w2v = W2V::new(5, 0.01);
    w2v.forward(input.view(), target.view()).unwrap();
    let label = Array1::from_vec(vec![1, 0, 1, 0, 1]);
    c.bench_function("backward", |b| b.iter(|| w2v.backward(label.clone())));
}
