use crate::word2vec::{Layer, sigmoid, binary_entropy_loss};
use ndarray::{ Array2, ArrayView2, Axis };


pub struct DocumentLayer {
    layer: Layer,
    lr: f32,
}

impl DocumentLayer {
    pub fn new(embedding_dim: usize, lr: f32) -> Self {
        let layer = Layer::new(embedding_dim);
        DocumentLayer { layer, lr }
    }

    pub fn forward(&self, input: ArrayView2<f32>, word_embedding: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>) {
        let doc_embedding = self.layer.forward(input);
        let sim: Vec<f32> = (doc_embedding.clone() * word_embedding).axis_iter(Axis(0))
        .map(|x| x.sum())
        .collect();
        (sigmoid(Array2::from_shape_vec((sim.len(), 1), sim).unwrap()), doc_embedding)
    }

    pub fn backward(&mut self, output: Array2<f32>, doc_vec: &Array2<f32>, input: &Array2<f32>) {
        let loss: Array2<f32> = binary_entropy_loss(Array2::<u32>::ones((output.len(), 1)), output.clone());
        let out_grad: Array2<f32> = output.clone() * (Array2::<f32>::ones((output.len(), 1)) - output); // output gradient
        let wr_z: Array2<f32> = loss.dot(&out_grad); // sigmoid gradient
        let doc_gradient: Array2<f32> = wr_z.dot(doc_vec); // gradient w.r.t. document vector
        let grad: Array2<f32> = doc_gradient.dot(&input.t());

        // update weights and biases
        self.layer.weights -= &(grad * self.lr);
        self.layer.biases -= &(doc_gradient * self.lr);
    }
}


#[cfg(test)]
mod tests {
    use::super::*;
    use word2vec::binary_entropy_loss;

    #[test]
    fn test_doc_layer_creation() {
        let doc_layer = DocumentLayer::new(3, 0.01);
        assert_eq!(doc_layer.layer.weights.shape(), (3, 3));
        assert_eq!(doc_layer.layer.biases.shape(), (3,));
        assert_eq!(doc_layer.lr, 0.01);
    }

    #[test]
    fn test_doc_layer_forward() {
        let doc_layer = DocumentLayer::new(3, 0.01);
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let word_embedding = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let (output, doc_vec) = doc_layer.forward(input.view(), word_embedding.view());
        assert_eq!(output.shape(), (2,));
        assert_eq!(doc_vec.shape(), (2, 3));
    }

    #[test]
    fn test_doc_layer_backward() {
        let doc_layer = DocumentLayer::new(3, 0.01);
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let word_embedding = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let (output, doc_vec) = doc_layer.forward(input.view(), word_embedding.view());
        let loss = binary_entropy_loss(Array2::ones((2, 1)), output.clone());
        doc_layer.backward(loss, doc_vec.view(), input.view());
        assert!(doc_layer.layer.weights.iter().all(|&x| x != 0.0));
    }
}
