use ndarray::{Array2, ArrayView2, Axis};
use ndarray_rand::rand_distr::Uniform;
use pyo3::prelude::*;
use rayon::prelude::*;


fn binary_entropy(target: Array2<u32>, pred: Array2<f32>)-> f32 {
    - (target * pred.ln()) + (1 - target) * (1 - pred).ln()
}

fn sigmoid(output: Array2<f32>) -> Array2<f32> {
    let base: f64 = std::f64::consts::E;
    output.mapv(|x| 1.0 / (1.0 + base.pow(-x)))
}

pub struct Layer {
    weights: Array2<f32>,
    biases: Array2<f32>,
}

impl Layer {
    fn new(embedding_dim: usize) -> Self {
        let weights = Array2::random((1, embedding_dim), Uniform::new(-0.1, 0.1));
        let biases = Array2::random((1, embedding_dim), Uniform::new(-0.1, 0.1));
        Layer { weights, biases }
    }

    fn forward(&self, input: ArrayView2<f32>) -> Array2<f32> {
        input.dot(&self.weights) + &self.biases
    }
}


#[pyclass]
pub struct Word2Vec {
    #[pyo3(get)]
    pub embedding_dim: usize,
    #[pyo3(get)]
    pub vocab_size: usize,
    pub input_layer: Layer,
    pub target_layer: Layer,
    #[pyo3(get)]
    pub lr: f32,
}

#[pymethods]
impl Word2Vec {
    #[new]
    fn new(embedding_dim: usize, lr: f32) -> Self {
        let input_layer = Layer::new(embedding_dim);
        let target_layer = Layer::new(embedding_dim);
        Word2Vec {
            embedding_dim,
            vocab_size,
            input_layer,
            target_layer,
            lr
        }
    }


    fn forward(&self, input: ArrayView2<usize>, target: ArrayView2<usize>) -> Array2<f32> {
        let input_output = self.input_layer.forward(input);
        let target_ouput = self.target_layer.forward(target);
        sigmoid(input_output.dot(&target_ouput.t()).diag());
    }

    fn backward(&self, output: Array2<f32>, label: ArrayView2<u32>) {
        // TODO: Fix this
        let loss = binary_entropy(label, output);
        let input_grad = loss.dot(&self.target_layer.weights.t());
        let target_grad = loss.dot(&self.input_layer.weights.t());

        // get gradients with respect to biases
        let input_bias_grad = loss.sum_axis(Axis(0));
        let target_bias_grad = loss.sum_axis(Axis(0));

        // update weights and biases
        self.input_layer.biases -= &(input_bias_grad * self.lr);
        self.target_layer.biases -= &(target_bias_grad * self.lr);

        self.input_layer.weights -= &(input_grad * self.lr);
        self.target_layer.weights -= &(target_grad * self.lr);
    }

    fn __call__(&self, input: Vec<usize>, target: Vec<usize>, label: Vec<u32>, grad: Option<bool>) -> PyResult<Vec<Vec<f32>>> {
        let input_array = Array2::from_shape_vec((1, input.len()), input).unwrap();
        let target_array = Array2::from_shape_vec((1, target.len()), target).unwrap();
        let label_array = Array2::from_shape_vec((1, label.len()), label).unwrap();

        let output = self.forward(input_array.view(), target_array.view());

        if grad.unwrap_or(false) {
            self.backward(output, label_array.view());
        }

        Ok(output.par_iter().map(|&x| x as f32).collect())
    }
}
