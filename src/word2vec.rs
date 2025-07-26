use ndarray::{Array, Array2, ArrayView2, Axis};
use ndarray_rand::rand_distr::Uniform;
use pyo3::prelude::*;
use ndarray_rand::RandomExt;


pub fn binary_entropy_loss(target: Array2<u32>, pred: Array2<f32>)-> Array2<f32> {
    let t = target.mapv(|x| x as f32);
    (pred.clone() - t.clone()) / (pred * (1.0 - t.clone()))
}

fn sigmoid(output: Array2<f32>) -> Array2<f32> {
    let base: f32 = std::f32::consts::E;
    output.mapv(|x| 1.0 / (1.0 + base.powf(-x)))
}

fn xavier_uniform(shape: (usize, usize)) -> Array2<f32> {
    let limit = (6.0 / (shape.0 + shape.1) as f32).sqrt();
    Array2::random(shape, Uniform::new(-limit, limit))
}

pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array2<f32>,
}

impl Layer {
    fn new(embedding_dim: usize) -> Self {
        let weights = xavier_uniform((1, embedding_dim));
        let biases = xavier_uniform((1, embedding_dim));
        Layer { weights, biases }
    }

    fn forward(&self, input: ArrayView2<f32>) -> Array2<f32> {
        input.dot(&self.weights) + &self.biases
    }
}


pub struct W2V {
    pub embedding_dim: usize,
    pub lr: f32,
    pub input_layer: Layer,
    pub target_layer: Layer
}


impl W2V {
    pub fn new(embedding_dim: usize, lr: f32) -> Self {
        let input_layer = Layer::new(embedding_dim);
        let target_layer = Layer::new(embedding_dim);
        W2V {
            embedding_dim,
            lr,
            input_layer,
            target_layer
        }
    }

    pub fn forward(&self, input: ArrayView2<f32>, target: ArrayView2<f32>) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>), String> {
        let input_output = self.input_layer.forward(input);
        let target_output = self.target_layer.forward(target);
        let consine_sim: Vec<f32> = (input_output.clone() * target_output.clone()).axis_iter(Axis(0))
            .map(|x| x.sum())
            .collect();
        Ok((sigmoid(Array2::from_shape_vec((consine_sim.len(), 1), consine_sim).unwrap()), input_output, target_output))
    }

    pub fn backward(&mut self, loss: Array2<f32>, input: Array2<f32>, target: Array2<f32>, output: Array2<f32>, y0: Array2<f32>, y1: Array2<f32>) -> Result<(), String> {
        // loss is the binary cross-entropy loss
        // input is the word vector for the input word
        // target is the context word vector
        // output is the sigmoid output of the dot product of input and target
        // y0 is the input word embedding
        // y1 is the target word embedding
        let shape = output.raw_dim();
        let wr_z = loss.dot(&(output.clone() * (Array::<f32, _>::ones(shape) - output))); // sigmoid gradient
        let input_bias_grad = wr_z.dot(&y1); // gradient w.r.t. input word embedding
        let target_bias_grad = wr_z.dot(&y0); // gradient w.r.t. target word embedding
        let input_grad = input_bias_grad.dot(&input.t()); // gradient w.r.t. input layer weights
        let target_grad = target_bias_grad.dot(&target.t()); // gradient w.r.t. target layer weights

        // update weights and biases
        self.input_layer.biases -= &(input_bias_grad * self.lr);
        self.target_layer.biases -= &(target_bias_grad * self.lr);
        self.input_layer.weights -= &(input_grad * self.lr);
        self.target_layer.weights -= &(target_grad * self.lr);
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_entropy_loss() {
        let target = Array2::from_shape_vec((1, 3), vec![1, 0, 1]).unwrap();
        let pred = Array2::from_shape_vec((1, 3), vec![0.9, 0.1, 0.8]).unwrap();
        let loss = binary_entropy_loss(target, pred);
    }

    #[test]
    fn test_sigmoid() {
        let input = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, -1.0]).unwrap();
        let output = sigmoid(input);
        assert_eq!(output.shape(), &[1, 3]);
        assert!(output.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(5);
        assert_eq!(layer.weights.shape(), &[1, 5]);
        assert_eq!(layer.biases.shape(), &[1, 5]);
    }

    #[test]
    fn test_layer_forward() {
        let layer = Layer::new(3);
        let input = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let output = layer.forward(input.view());
        assert_eq!(output.shape(), &[3, 3]);
    }

    #[test]
    fn test_w2v_creation() {
        let w2v = W2V::new(10, 0.01);
        assert_eq!(w2v.embedding_dim, 10);
        assert_eq!(w2v.lr, 0.01);
        assert_eq!(w2v.input_layer.weights.shape(), &[1, 10]);
        assert_eq!(w2v.target_layer.weights.shape(), &[1, 10]);
    }

    #[test]
    fn test_w2v_forward() {
        let w2v = W2V::new(5, 0.01);
        let input = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let target = Array2::from_shape_vec((1, 5), vec![5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
        let (output, input_output, target_output) = w2v.forward(input.view(), target.view()).unwrap();
        assert_eq!(output.shape(), &[1, 5]);
        assert_eq!(input_output.shape(), &[1, 5]);
        assert_eq!(target_output.shape(), &[1, 5]);
    }
}
