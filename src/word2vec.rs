extern crate blas_src;
extern crate ndarray;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::utils::{binary_entropy_grad, sigmoid, Layer};

pub struct W2V {
    pub embedding_dim: usize,
    pub lr: f32,
    pub input_layer: Layer,
    pub context_layer: Layer,
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
        }
    }

    pub fn train_batch(
        &mut self,
        input: ArrayView2<f32>,
        context: ArrayView2<f32>,
        y_true: ArrayView1<f32>,
    ) -> Result<(), String> {
        // Forward pass
        let input_embedding = self.input_layer.forward(input);
        let context_embedding = self.context_layer.forward(context);

        let consine_sim = (0..input_embedding.shape()[0])
            .map(|i| {
                let input_vec = input_embedding.row(i);
                let context_vec = context_embedding.row(i);
                input_vec.dot(&context_vec)
            })
            .collect::<Array1<f32>>();

        let sig = sigmoid(consine_sim);

        // Backward pass
        let loss: f32 = binary_entropy_grad(y_true, sig);
        let context_embedding_grad = loss * input_embedding; // gradient w.r.t. context word embedding
        let input_embedding_grad = loss * context_embedding; // gradient w.r.t. input word embedding
        let input_grad = input.t().dot(&input_embedding_grad); // gradient w.r.t. input layer weights
        let target_grad = context.t().dot(&context_embedding_grad); // gradient w.r.t. target layer weights

        // update weights
        self.input_layer.weights -= &(input_grad * self.lr);
        self.context_layer.weights -= &(target_grad * self.lr);
        Ok(())
    }

    pub fn predict(&mut self, input: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        Ok(self.input_layer.forward(input))
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
    fn test_train_batch() {
        let mut w2v = W2V::new(5, 0.01);
        let input = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let context = Array2::from_shape_vec((3, 1), vec![4.0, 5.0, 6.0]).unwrap();
        let y_true = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let result = w2v.train_batch(input.view(), context.view(), y_true.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_predict() {
        let mut w2v = W2V::new(5, 0.01);
        let input = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let context = Array2::from_shape_vec((3, 1), vec![4.0, 5.0, 6.0]).unwrap();
        let y_true = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let _ = w2v
            .train_batch(input.view(), context.view(), y_true.view())
            .unwrap();
        let prediction = w2v.predict(input.view()).unwrap();
        assert_eq!(prediction.shape(), &[3, 5]);
    }
}
