mod word2vec;

use word2vec::{Layer, sigmoid, binary_entropy};
use pyo3::prelude::*;
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;

#[pyclass]
struct Doc2Vec {
    embedding_dim: usize,
    vocab_size: usize,
    input_layer: Layer,
    target_layer: Layer,
    doc_layer: Layer,
    lr: f32,
}

#[pymethods]
impl Doc2Vec {
    #[new]
    fn new(embedding_dim: usize, vocab_size: usize, lr: f32) -> Self {
        let input_layer = Layer::new(embedding_dim);
        let target_layer = Layer::new(embedding_dim);
        let doc_layer = Layer::new(embedding_dim);
        Doc2Vec {
            embedding_dim,
            vocab_size,
            input_layer,
            target_layer,
            doc_layer,
            lr,
        }
    }

    fn forward(&self, input: ArrayView2<usize>, target: ArrayView2<usize>, doc: ArrayView2<usize>) -> Array2<f32> {
        let input_embed = self.input_layer.forward(input);
        let target_embed = self.target_layer.forward(target);
        let doc_embed = self.doc_layer.forward(doc);
        let combined = (doc_embeddings + input_embeddings) / 2;
        sigmoid(combined.dot(&target_embed.t()).diag());
    }


    fn backward(&self, output: Array2<f32>, label: ArrayView2<u32>) {
        // TODO: Fix this
        let loss = binary_entropy(label, output);
        let input_grad = loss.dot(&self.target_layer.weights.t());
        let target_grad = loss.dot(&self.input_layer.weights.t());
        let doc_grad = loss.dot(&self.doc_layer.weights.t());

        // get gradients with respect to biases
        let input_bias_grad = loss.sum_axis(Axis(0));
        let target_bias_grad = loss.sum_axis(Axis(0));
        let doc_bias_grad = loss.sum_axis(Axis(0));

        // update weights and biases
        self.input_layer.biases -= &(input_bias_grad * self.lr);
        self.target_layer.biases -= &(target_bias_grad * self.lr);
        self.doc_layer.biases -= &(doc_bias_grad * self.lr);

        self.input_layer.weights -= &(input_grad * self.lr);
        self.target_layer.weights -= &(target_grad * self.lr);
        self.doc_layer.weights -= &(doc_grad * self.lr);
    }


    fn __call__(&self, input: Vec<usize>, target: Vec<usize>, doc: Vec<usize>, label: Vec<usize>, grad: Option<bool>) -> PyResult<Vec<Vec<f32>>> {
        let input_array = Array2::from_shape_vec((1, input.len()), input).unwrap();
        let target_array = Array2::from_shape_vec((1, target.len()), target).unwrap();
        let doc_array = Array2::from_shape_vec((1, doc.len()), doc).unwrap();
        let label_array = Array2::from_shape_vec((1, label.len()), label).unwrap();

        let out = self.forward(input, target, doc);
        if grad.unwrap_or(false) {
            self.backward(out.clone(), label);
        }

        Ok(out.par_iter().map(|&x| x as f32).collect())
    }

    fn infer_vectors(&self, doc: Vec<usize>) -> PyResult<Vec<f32>> {
        // let doc_array = Array2::from_shape_vec((1, doc.len()), doc).unwrap();
        // let doc_embed = self.doc_layer.forward(doc_array.view());
        // Ok(doc_embed.par_iter().map(|&x| x as f32).collect())
    }
}
