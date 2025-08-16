extern crate blas_src;
extern crate ndarray;

use crate::utils::{binary_entropy_grad, sigmoid, Layer};
use ndarray::{Array1, Array2, ArrayView2, Axis};

pub struct DocumentLayer {
    layer: Layer,
    lr: f32,
}

impl DocumentLayer {
    pub fn new(embedding_dim: usize, lr: f32) -> Self {
        let layer = Layer::new(embedding_dim);
        DocumentLayer { layer, lr }
    }

    pub fn train_batch(
        &mut self,
        input: ArrayView2<f32>,
        word_embedding: ArrayView2<f32>,
    ) -> Result<(), String> {
        let num_samples = input.shape()[0];
        // forward pass
        let doc_embedding = self.layer.forward(input);
        let sim = (doc_embedding * word_embedding).sum_axis(Axis(1)); // this could likely be sped up
        let sig = sigmoid(sim);

        // backward pass
        let loss: f32 = binary_entropy_grad(Array1::ones(num_samples).view(), sig);
        let doc_loss = loss * &word_embedding;
        let doc_grad = doc_loss.t().dot(&input); // gradient w.r.t. input layer weights

        // update weights
        self.layer.weights -= &(doc_grad * self.lr).t();

        Ok(())
    }

    pub fn predict(&mut self, input: ArrayView2<f32>) -> Result<Array2<f32>, String> {
        Ok(self.layer.forward(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::binary_entropy_loss;

    #[test]
    fn test_doc_layer_creation() {
        let doc_layer = DocumentLayer::new(3, 0.01);
        assert_eq!(doc_layer.layer.weights.shape(), &[1, 3]);
        assert_eq!(doc_layer.lr, 0.01);
    }

    #[test]
    fn test_doc_layer_train_batch() {
        let mut doc_layer = DocumentLayer::new(3, 0.01);
        let input = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let word_embedding =
            Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
        let training_res = doc_layer.train_batch(input.view(), word_embedding.view());
        assert_eq!(training_res.is_ok(), true);

        assert_eq!(doc_layer.layer.weights.shape(), &[1, 3]);
    }

    #[test]
    fn test_doc_layer_predict() {
        let mut doc_layer = DocumentLayer::new(3, 0.01);
        let input = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let word_embedding =
            Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
        doc_layer
            .train_batch(input.view(), word_embedding.view())
            .expect("Training failed");

        let prediction = doc_layer.predict(input.view()).expect("Prediction failed");
        assert_eq!(prediction.shape(), &[2, 3]);
    }
}
