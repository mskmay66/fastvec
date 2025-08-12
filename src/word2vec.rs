extern crate ndarray;

use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::collections::HashMap;

use crate::utils::{binary_entropy_grad, sigmoid, GradVars, Layer};

pub struct W2V {
    pub embedding_dim: usize,
    pub lr: f32,
    pub input_layer: Layer,
    pub context_layer: Layer,
    grad_vars: HashMap<String, GradVars>,
}

impl W2V {
    pub fn new(embedding_dim: usize, lr: f32) -> Self {
        let input_layer = Layer::new(embedding_dim);
        let context_layer = Layer::new(embedding_dim);
        W2V {
            embedding_dim,
            lr,
            input_layer,
            context_layer,
            grad_vars: HashMap::new(),
        }
    }

    pub fn forward(
        &mut self,
        input: ArrayView2<f32>,
        context: ArrayView2<f32>,
    ) -> Result<Array2<f32>, String> {
        // add input and target to grad_vars
        self.grad_vars
            .insert("input".to_string(), GradVars::Arr2(input.to_owned()));
        self.grad_vars
            .insert("context".to_string(), GradVars::Arr2(context.to_owned()));

        let input_embedding = self.input_layer.forward(input);
        let context_embedding = self.context_layer.forward(context);

        // add embedding outputs to grad_vars
        self.grad_vars.insert(
            "input_embedding".to_string(),
            GradVars::Arr2(input_embedding.clone()),
        );
        self.grad_vars.insert(
            "context_embedding".to_string(),
            GradVars::Arr2(context_embedding.clone()),
        );

        let consine_sim = (0..input_embedding.shape()[0])
            .map(|i| {
                let input_vec = input_embedding.row(i);
                let context_vec = context_embedding.row(i);
                input_vec.dot(&context_vec)
            })
            .collect::<Array1<f32>>();

        // add cosine similarity to grad_vars
        self.grad_vars.insert(
            "consine_sim".to_string(),
            GradVars::Arr1(consine_sim.clone()),
        );
        // apply sigmoid to cosine similarity
        let sig = sigmoid(consine_sim.clone());
        // add sigmoid output to grad_vars
        self.grad_vars
            .insert("sigmoid_output".to_string(), GradVars::Arr1(sig.clone()));
        Ok(input_embedding)
    }

    pub fn backward(&mut self, y_true: Array1<u32>) -> Result<(), String> {
        let loss: Array2<f32> =
            binary_entropy_grad(y_true, self.grad_vars["sigmoid_output"].unwrap_arr1())
                .insert_axis(Axis(1)); // gradient of binary cross-entropy loss
        let context_sum = self.grad_vars["context_embedding"]
            .unwrap_arr2()
            .sum_axis(Axis(0)); // sum over all context embeddings
        let input_sum = self.grad_vars["input_embedding"]
            .unwrap_arr2()
            .sum_axis(Axis(0)); // sum over all input embeddings

        let input_before_weights = loss.dot(&context_sum.clone().insert_axis(Axis(0))); // gradient w.r.t. input word embedding
        let target_before_weights = loss.dot(&input_sum.clone().insert_axis(Axis(0))); // gradient w.r.t. target word embedding
        let input = self.grad_vars["input"].unwrap_arr2();
        let context = self.grad_vars["context"].unwrap_arr2();

        let it = input.t(); // copying in memory could make this slow
        let ct = context.t();

        let input_grad = it.dot(&input_before_weights); // gradient w.r.t. input layer weights
        let target_grad = ct.dot(&target_before_weights); // gradient w.r.t. target layer weights

        // update weights and biases
        self.input_layer.biases -= &(context_sum * self.lr);
        self.context_layer.biases -= &(input_sum * self.lr);
        self.input_layer.weights -= &(input_grad * self.lr);
        self.context_layer.weights -= &(target_grad * self.lr);

        self.grad_vars.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::binary_entropy_loss;

    #[test]
    fn test_w2v_creation() {
        let w2v = W2V::new(10, 0.01);
        assert_eq!(w2v.embedding_dim, 10);
        assert_eq!(w2v.lr, 0.01);
        assert_eq!(w2v.input_layer.weights.shape(), &[1, 10]);
        assert_eq!(w2v.context_layer.weights.shape(), &[1, 10]);
    }

    #[test]
    fn test_w2v_forward() {
        let mut w2v = W2V::new(5, 0.01);
        let input = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let target = Array2::from_shape_vec((5, 1), vec![5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
        let word_embedding = w2v.forward(input.view(), target.view()).unwrap();
        assert_eq!(word_embedding.shape(), &[5, 5]);

        assert_eq!(
            w2v.grad_vars.get("input").unwrap().unwrap_arr2().shape(),
            &[5, 1]
        );
        assert_eq!(
            w2v.grad_vars.get("context").unwrap().unwrap_arr2().shape(),
            &[5, 1]
        );
        assert_eq!(
            w2v.grad_vars
                .get("input_embedding")
                .unwrap()
                .unwrap_arr2()
                .shape(),
            &[5, 5]
        );
        assert_eq!(
            w2v.grad_vars
                .get("context_embedding")
                .unwrap()
                .unwrap_arr2()
                .shape(),
            &[5, 5]
        );
        assert_eq!(
            w2v.grad_vars
                .get("consine_sim")
                .unwrap()
                .unwrap_arr1()
                .shape(),
            &[5]
        );
        assert_eq!(
            w2v.grad_vars
                .get("sigmoid_output")
                .unwrap()
                .unwrap_arr1()
                .shape(),
            &[5]
        );
    }

    #[test]
    fn test_w2v_backward() {
        let mut w2v = W2V::new(5, 0.01);
        let input = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let target = Array2::from_shape_vec((6, 1), vec![5.0, 4.0, 3.0, 2.0, 1.0, 1.0]).unwrap();
        let _ = w2v.forward(input.view(), target.view()).unwrap();
        let y_true = Array1::from_vec(vec![1, 0, 1, 0, 1, 1]);
        let result = w2v.backward(y_true);
        assert!(result.is_ok());
        assert_eq!(w2v.input_layer.biases.shape(), &[1, 5]);
        assert_eq!(w2v.context_layer.biases.shape(), &[1, 5]);
        assert_eq!(w2v.input_layer.weights.shape(), &[1, 5]);
        assert_eq!(w2v.context_layer.weights.shape(), &[1, 5]);
    }

    #[test]
    fn test_loss_decreasing() {
        let mut w2v = W2V::new(5, 0.001);
        let input = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let target = Array2::from_shape_vec((5, 1), vec![5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
        let y_true = Array1::from_vec(vec![0, 1, 0, 0, 1]);
        for _ in 0..5 {
            let _ = w2v.forward(input.view(), target.view()).unwrap();
            let loss = binary_entropy_loss(
                y_true.clone(),
                w2v.grad_vars["sigmoid_output"].unwrap_arr1(),
            );
            println!("Current loss: {}", loss);
            let _ = w2v.backward(y_true.clone());
        }
    }
}
