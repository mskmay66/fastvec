mod word2vec;

use word2vec::{Layer, sigmoid, binary_entropy_loss};
use pyo3::prelude::*;
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;


#[pyclass]
struct DocumentLayer {
    layer: Layer,
    lr: f32,
}

#[pymethods]
impl DocumentLayer {
    #[new]
    fn new(embedding_dim: usize, lr: f32) -> Self {
        let layer = Layer::new(embedding_dim);
        DocumentLayer { layer, lr }
    }

    fn forward(&self, input: ArrayView2<f32>) -> Array2<f32> {
        self.layer.forward(input)
    }

    fn backward(&self, loss: Array2<f32>, doc_vec: ArrayView2<f32>, input: ArrayView2<f32>) {
        let wr_z = loss.dot(output * (1 - output)); // sigmoid gradient
        let doc_gradient = wr_z.dot(doc_vec); // gradient w.r.t. input word embedding
        let grad = doc_gradient.dot(&input.t());
        self.layer.weights -= &(grad * lr);
        self.layer.biases -= &(doc_gradient * self.lr);
    }

    pub fn infer_vectors(&self, input: ArrayView2, word_vector: ArrayView2, epochs: Option<u32>) -> PyResult<Array2<f32>> {
        // start with random embedding vector
        let mut embedding = Array2::random((1, self.layer.weights.shape()[1]), Uniform::new(-1.0, 1.0));
        let mut prev_embedding = embedding.clone();
        let mut sim = 0;
        for _ in 0..epochs {
            embedding = self.forward(embedding);
            sim = (input_output * target_output).par_iter().sum::<f32>();
            loss = binary_entropy_loss(sigmoid(sim), Array2::ones((1, input.len())));
            self.backward(loss, embedding.view(), prev_embedding.view());
            prev_embedding = embedding.clone();
        }
        Ok(embedding)
    }
}
