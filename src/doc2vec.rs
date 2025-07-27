mod word2vec;

use word2vec::{Layer, sigmoid, binary_entropy_loss};
use ndarray::{Array2, ArrayView2, Axis};


struct DocumentLayer {
    layer: Layer,
    lr: f32,
}

impl DocumentLayer {
    fn new(embedding_dim: usize, lr: f32) -> Self {
        let layer = Layer::new(embedding_dim);
        DocumentLayer { layer, lr }
    }

    fn forward(&self, input: ArrayView2<f32>, word_embedding: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>) {
        let doc_embedding = self.layer.forward(input);
        let sim = (doc_embedding * word_embedding).Axis(0).sum();
        (sigmoid(sim), doc_embedding)
    }

    fn backward(&self, loss: Array2<f32>, doc_vec: ArrayView2<f32>, input: ArrayView2<f32>) {
        let wr_z = loss.dot(output * (Array::<f32, _>::ones(shape) - output)); // sigmoid gradient
        let doc_gradient = wr_z.dot(doc_vec); // gradient w.r.t. input word embedding
        let grad = doc_gradient.dot(&input.t());
        self.layer.weights -= &(grad * lr);
        self.layer.biases -= &(doc_gradient * self.lr);
    }
}
