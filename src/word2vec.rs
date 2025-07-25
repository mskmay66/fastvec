use ndarray::{Array2, ArrayView2, Axis};
use ndarray_rand::rand_distr::Uniform;
use pyo3::prelude::*;
use rayon::prelude::*;


fn binary_entropy_loss(target: Array2<u32>, pred: Array2<f32>)-> f32 {
    (pred - target) / (pred * (1 - target))
}

fn sigmoid(output: Array2<f32>) -> Array2<f32> {
    let base: f64 = std::f64::consts::E;
    output.mapv(|x| 1.0 / (1.0 + base.pow(-x)))
}

pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array2<f32>,
}

impl Layer {
    fn xavier_uniform(shape: (usize, usize)) -> Array2<f32> {
        let limit = (6.0 / (shape.0 + shape.1) as f32).sqrt();
        Array2::random(shape, Uniform::new(-limit, limit))
    }

    fn new(embedding_dim: usize) -> Self {
        let weights = xavier_uniform(1, embedding_dim);
        let biases = xavier_uniform(1, embedding_dim);
        Layer { weights, biases }
    }

    fn forward(&self, input: ArrayView2<f32>) -> Array2<f32> {
        input.dot(&self.weights) + &self.biases
    }
}


#[pyclass]
pub struct _Word2Vec {
    #[pyo3(get)]
    pub embedding_dim: usize,
    #[pyo3(get)]
    pub vocab_size: usize,
    #[pyo3(get)]
    pub lr: f32,
    pub input_layer: Layer,
    pub target_layer: Layer
}

#[pymethods]
impl _Word2Vec {
    #[new]
    fn new(embedding_dim: usize, lr: f32) -> Self {
        let input_layer = Layer::new(embedding_dim);
        let target_layer = Layer::new(embedding_dim);
        _Word2Vec {
            embedding_dim,
            vocab_size,
            lr,
            input_layer,
            target_layer
        }
    }


    fn forward(&self, input: ArrayView2<usize>, target: ArrayView2<usize>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let input_output = self.input_layer.forward(input);
        let target_ouput = self.target_layer.forward(target);
        let consine_sim = (input_output * target_output).par_iter().sum::<f32>();
        (sigmoid(consine_sim) , input_output, target_output)
    }

    fn backward(&self, loss: Array2<f32>, input: Array2<f32>, target: Array2<f32>, output: Array2<f32>, y0: Array2<f32>, y1: Array2<f32>) -> PyResult<()> {
        // loss is the binary cross-entropy loss
        // input is the word vector for the input word
        // target is the context word vector
        // output is the sigmoid output of the dot product of input and target
        // y0 is the input word embedding
        // y1 is the target word embedding
        let wr_z = loss.dot(output * (1 - output)); // sigmoid gradient
        let input_bias_grad = wr_z.dot(y1); // gradient w.r.t. input word embedding
        let target_bias_grad = wr_z.dot(y0); // gradient w.r.t. target word embedding
        let input_grad = input_bias_grad.dot(input.t()); // gradient w.r.t. input layer weights
        let target_grad = target_bias_grad.dot(target.t()); // gradient w.r.t. target layer weights

        // update weights and biases
        self.input_layer.biases -= &(input_bias_grad * self.lr);
        self.target_layer.biases -= &(target_bias_grad * self.lr);
        self.input_layer.weights -= &(input_grad * self.lr);
        self.target_layer.weights -= &(target_grad * self.lr);
        Ok()
    }

    fn __call__(&self, input: Vec<usize>, target: Vec<usize>, label: Vec<u32>, grad: Option<bool>) -> PyResult<Vec<Vec<f32>>> {
        let input_array = Array2::from_shape_vec((1, input.len()), input).unwrap();
        let target_array = Array2::from_shape_vec((1, target.len()), target).unwrap();
        let label_array = Array2::from_shape_vec((1, label.len()), label).unwrap();

        let (output, y0, y1) = self.forward(input_array.view(), target_array.view());

        if grad.unwrap_or(false) {
            let loss = binary_entropy_loss(label_array.view(), output.view());
            self.backward(loss, input_array, target_array, output, y0, y1);
        }

        Ok(output.par_iter().map(|&x| x as f32).collect())
    }
}
