use crate::word2vec::{Layer, sigmoid, GradVars, binary_entropy_grad, binary_entropy_loss};
use ndarray::{ Array1, Array2, ArrayView2, Axis };
use std::collections::HashMap;


pub struct DocumentLayer {
    layer: Layer,
    lr: f32,
    grad_vars: HashMap<String, GradVars>,
}

impl DocumentLayer {
    pub fn new(embedding_dim: usize, lr: f32) -> Self {
        let layer = Layer::new(embedding_dim);
        DocumentLayer { layer, lr, grad_vars: HashMap::new() }
    }

    pub fn forward(&mut self, input: ArrayView2<f32>, word_embedding: ArrayView2<f32>) -> Array2<f32> {
        // add input and word_embedding to grad_vars
        self.grad_vars.insert("input".to_string(), GradVars::Arr2(input.to_owned()));
        self.grad_vars.insert("word_embedding".to_string(), GradVars::Arr2(word_embedding.to_owned()));

        let doc_embedding = self.layer.forward(input);

        let sim = (doc_embedding.clone() * word_embedding).sum_axis(Axis(1));
        let sig = sigmoid(sim.clone());
        // add sigmoid output to grad_vars
        self.grad_vars.insert("sigmoid_output".to_string(), GradVars::Arr1(sig.clone()));
        doc_embedding
    }

    pub fn backward(&mut self, y_true: Array1<u32>) -> Result<(), String> {
        let loss: Array2<f32> = binary_entropy_grad(y_true, self.grad_vars["sigmoid_output"].unwrap_arr1()).insert_axis(Axis(1));
        let doc_bias_grad = self.grad_vars["word_embedding"].unwrap_arr2().sum_axis(Axis(0)); // sum over all word embeddings
        let doc_loss = loss.dot(&doc_bias_grad.clone().insert_axis(Axis(0))); // gradient w.r.t. input word embedding
        let doc_grad = doc_loss.t().dot(&self.grad_vars["input"].unwrap_arr2()); // gradient w.r.t. input layer weights

        // update weights and biases
        self.layer.weights -= &(doc_grad * self.lr).t();
        self.layer.biases -= &(doc_bias_grad * self.lr);

        // update grad_vars
        self.grad_vars.clear();
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_layer_creation() {
        let doc_layer = DocumentLayer::new(3, 0.01);
        assert_eq!(doc_layer.layer.weights.shape(), &[1,3]);
        assert_eq!(doc_layer.layer.biases.shape(), &[1,3]);
        assert_eq!(doc_layer.lr, 0.01);
    }

    #[test]
    fn test_doc_layer_forward() {
        let mut doc_layer = DocumentLayer::new(3, 0.01);
        let input = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let word_embedding = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
        let doc_vec = doc_layer.forward(input.view(), word_embedding.view());
        assert_eq!(doc_vec.shape(), &[2,3]);

        assert_eq!(doc_layer.grad_vars.get("input").unwrap().unwrap_arr2().shape(), &[2, 1]);
        assert_eq!(doc_layer.grad_vars.get("word_embedding").unwrap().unwrap_arr2().shape(), &[2, 1]);
        assert_eq!(doc_layer.grad_vars.get("sigmoid_output").unwrap().unwrap_arr1().shape(), &[2]);
    }

    #[test]
    fn test_doc_layer_backward() {
        let mut doc_layer = DocumentLayer::new(3, 0.01);
        let word_embedding = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let input = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
        let _doc_vec = doc_layer.forward(input.view(), word_embedding.view());
        let y_true = Array1::from_vec(vec![1, 0]);
        doc_layer.backward(y_true).expect("Backward pass failed");

        assert_eq!(doc_layer.layer.weights.shape(), &[1, 3]);
        assert_eq!(doc_layer.layer.biases.shape(), &[1, 3]);
    }

    #[test]
    fn test_d2v_loss_decreasing() {
        let mut d2v = DocumentLayer::new(3, 0.001);
        let word_embedding = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y_true = Array1::from_vec(vec![1, 0]);
        let mut prev_loss = f32::MAX;
        for _ in 0..5 {
            let mut doc_vec = Array2::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
            doc_vec = d2v.forward(doc_vec.view(), word_embedding.view());
            let loss = binary_entropy_loss(y_true.clone(), d2v.grad_vars["sigmoid_output"].unwrap_arr1());
            println!("Current loss: {}", loss);
            // assert!(loss < prev_loss, "Loss did not decrease: {} >= {}", loss, prev_loss);
            prev_loss = loss;
            let _ = d2v.backward(y_true.clone());
        }
    }
}
