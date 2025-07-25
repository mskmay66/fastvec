mod word2vec;

use word2vec::{Layer, sigmoid, binary_entropy_loss};
use pyo3::prelude::*;
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand_distr::Uniform;
use rayon::prelude::*;

#[pyclass]
struct _Doc2Vec {
    embedding_dim: usize,
    vocab_size: usize,
    input_layer: Layer,
    target_layer: Layer,
    doc_layer: Layer,
    lr: f32,
}

#[pymethods]
impl _Doc2Vec {
    #[new]
    fn new(embedding_dim: usize, vocab_size: usize, lr: f32) -> Self {
        let input_layer = Layer::new(embedding_dim);
        let target_layer = Layer::new(embedding_dim);
        let doc_layer = Layer::new(embedding_dim);
        _Doc2Vec {
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


    fn backward(&self, loss: Array2<f32>, input: Array2<f32>, target: Array2<f32>, output: Array2<f32>, y0: Array2<f32>, y1: Array2<f32>, y3: Array2<f32>) {
        // let base = loss.dot()
        let wr_z = loss.dot(output * (1 - output)); // sigmoid gradient
        let input_doc_avg_bias_grad = wr_z.dot(y1); // gradient w.r.t. input word embedding
        let target_bias_grad = wr_z.dot((y0 + y3) / 2); // gradient w.r.t. target word embedding
        let target_grad = target_bias_grad.dot(target.t());
        let input_grad = input_doc_avg_bias_grad.dot(input.t()) * 0.5; // gradient w.r.t. input layer weights
        let doc_grad = input_doc_avg_bias_grad.dot(doc.t()) * 0.5; // gradient w.r.t. doc layer weights

        // update weights and biases
        self.input_layer.biases -= &(input_doc_avg_bias_grad * self.lr * 0.5);
        self.doc_layer.biases -= &(input_doc_avg_bias_grad * self.lr * 0.5);
        self.target_layer.biases -= &(target_bias_grad * self.lr);
        self.doc_layer.biases -= &(doc_grad * self.lr);
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
