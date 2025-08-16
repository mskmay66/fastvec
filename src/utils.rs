use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub fn binary_entropy_grad(target: Array1<u32>, pred: Array1<f32>) -> f32 {
    (pred - target.mapv(|x| x as f32))
        .mean_axis(Axis(0))
        .unwrap()
        .into_scalar()
        .max(0.0) // ensure non-negative gradient
}

pub fn binary_entropy_loss(target: Array1<u32>, pred: Array1<f32>) -> f32 {
    let epsilon = 1e-15; // to avoid log(0)
    let t = target.mapv(|x| x as f32);
    let pred_clipped = pred.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
    let loss = -t.clone() * pred_clipped.mapv(|x| x.ln())
        - (1.0 - t) * (1.0 - pred_clipped).mapv(|x| x.ln());
    loss.mean_axis(Axis(0)).unwrap().into_scalar().max(0.0) // ensure non-negative loss
}

pub fn sigmoid(output: Array1<f32>) -> Array1<f32> {
    let base: f32 = std::f32::consts::E;
    output.mapv(|x| 1.0 / (1.0 + base.powf(-x)))
}

fn xavier_uniform(shape: (usize, usize)) -> Array2<f32> {
    let limit = (6.0 / (shape.0 + shape.1) as f32).sqrt();
    Array2::random(shape, Uniform::new(-limit, limit))
}

pub struct Layer {
    pub weights: Array2<f32>,
}

impl Layer {
    pub fn new(embedding_dim: usize) -> Self {
        let weights = xavier_uniform((1, embedding_dim));
        Layer { weights }
    }

    pub fn forward(&self, input: ArrayView2<f32>) -> Array2<f32> {
        input.dot(&self.weights)
    }
}

#[cfg(test)]
#[test]
fn test_loss() {
    let y_true = Array1::from_vec(vec![1, 0, 1, 0, 1]);
    let y_pred = Array1::from_vec(vec![0.9, 0.1, 0.8, 0.2, 0.7]);
    let loss = binary_entropy_loss(y_true, y_pred);
    println!("Loss: {}", loss);
    assert!(loss >= 0.0);
    assert!(loss < f32::MAX);
    assert!(loss.is_finite(), "Loss is not finite: {}", loss);
}

#[test]
fn test_sigmoid() {
    let input = Array1::from_vec(vec![0.0, 1.0, -1.0]);
    let output = sigmoid(input);
    assert!(output.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[test]
fn test_layer_creation() {
    let layer = Layer::new(5);
    assert_eq!(layer.weights.shape(), &[1, 5]);
}

#[test]
fn test_layer_forward() {
    let layer = Layer::new(3);
    let input = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let output = layer.forward(input.view());
    assert_eq!(output.shape(), &[3, 3]);

    let layer2 = Layer::new(5);
    let input2 = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let output2 = layer2.forward(input2.view());
    assert_eq!(output2.shape(), &[5, 5]);
}
