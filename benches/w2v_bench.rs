use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use std::hint::black_box;

use fastvec::builder::TrainingSet;
use fastvec::train_word2vec;
use fastvec::word2vec::W2V;

fn train_batch_benchmark(c: &mut Criterion) {
    let input = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let target = Array2::from_shape_vec((5, 1), vec![5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    let label = Array1::from_vec(vec![1, 0, 1, 0, 1]);
    let mut w2v = W2V::new(5, 0.01);
    c.bench_function("train_batch", |b| {
        b.iter(|| {
            w2v.train_batch(
                black_box(input.view()),
                black_box(target.view()),
                black_box(label.clone()),
            )
        })
    });
}

fn predict_benchmark(c: &mut Criterion) {
    let input = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let target = Array2::from_shape_vec((5, 1), vec![5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    let mut w2v = W2V::new(5, 0.01);
    c.bench_function("predict_word2vec", |b| {
        b.iter(|| w2v.predict(black_box(input.view())))
    });
}

fn train_benchmark(c: &mut Criterion) {
    let training_set = TrainingSet::new(
        vec![1, 2, 3, 4, 5],
        vec![2, 3, 4, 5, 6],
        vec![1, 1, 1, 1, 1],
        Some(5),
    );
    c.bench_function("train_word2vec", |b| {
        b.iter(|| {
            train_word2vec(
                black_box(training_set.clone()),
                black_box(5),
                black_box(0.01),
                black_box(10),
            )
        })
    });
}

criterion_group!(
    benches,
    train_batch_benchmark,
    train_benchmark,
    predict_benchmark
);
criterion_main!(benches);
