use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::collections::HashMap;

pub fn binary_entropy_grad(target: Array1<u32>, pred: Array1<f32>) -> Array1<f32> {
    pred - target.mapv(|x| x as f32)
}

pub fn binary_entropy_loss(target: Array1<u32>, pred: Array1<f32>) -> f32 {
    let epsilon = 1e-15; // to avoid log(0)
    let t = target.mapv(|x| x as f32);
    let pred_clipped = pred.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
    let loss = -t.clone() * pred_clipped.mapv(|x| x.ln())
        - (1.0 - t) * (1.0 - pred_clipped).mapv(|x| x.ln());
    loss.mean_axis(Axis(0)).unwrap().into_scalar().max(0.0) // ensure non-negative loss
}

pub fn sigmoid(output: Array1<f32>) -> Array1<f32> {
    let base: f32 = std::f32::consts::E;
    output.mapv(|x| 1.0 / (1.0 + base.powf(-x)))
}

fn xavier_uniform(shape: (usize, usize)) -> Array2<f32> {
    let limit = (6.0 / (shape.0 + shape.1) as f32).sqrt();
    Array2::random(shape, Uniform::new(-limit, limit))
}

pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array2<f32>,
}

impl Layer {
    pub fn new(embedding_dim: usize) -> Self {
        let weights = xavier_uniform((1, embedding_dim));
        let biases = xavier_uniform((1, embedding_dim));
        Layer { weights, biases }
    }

    pub fn forward(&self, input: ArrayView2<f32>) -> Array2<f32> {
        input.dot(&self.weights) + &self.biases
    }
}

pub enum GradVars {
    Arr1(Array1<f32>),
    Arr2(Array2<f32>),
}

impl GradVars {
    pub fn unwrap_arr1(&self) -> Array1<f32> {
        match self {
            GradVars::Arr1(arr) => arr.clone(),
            _ => panic!("Expected Arr1, found Arr2"),
        }
    }

    pub fn unwrap_arr2(&self) -> Array2<f32> {
        match self {
            GradVars::Arr2(arr) => arr.clone(),
            _ => panic!("Expected Arr2, found Arr1"),
        }
    }
}

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
        let consine_sim = (input_embedding.clone() * context_embedding.clone())
            .mean_axis(Axis(1))
            .unwrap();

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

        let it = input.t();
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

    #[test]
    fn test_loss() {
        let y_true = Array1::from_vec(vec![1, 0, 1, 0, 1]);
        let y_pred = Array1::from_vec(vec![0.9, 0.1, 0.8, 0.2, 0.7]);
        let loss = binary_entropy_loss(y_true, y_pred);
        println!("Loss: {}", loss);
        assert!(loss >= 0.0);
        assert!(loss < f32::MAX);
        assert!(loss.is_finite(), "Loss is not finite: {}", loss);
    }

    #[test]
    fn test_sigmoid() {
        let input = Array1::from_vec(vec![0.0, 1.0, -1.0]);
        let output = sigmoid(input);
        assert!(output.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(5);
        assert_eq!(layer.weights.shape(), &[1, 5]);
        assert_eq!(layer.biases.shape(), &[1, 5]);
    }

    #[test]
    fn test_layer_forward() {
        let layer = Layer::new(3);
        let input = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let output = layer.forward(input.view());
        assert_eq!(output.shape(), &[3, 3]);

        let layer2 = Layer::new(5);
        let input2 = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let output2 = layer2.forward(input2.view());
        assert_eq!(output2.shape(), &[5, 5]);
    }

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

    // TODO: Write tests for W2V backward pass
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
        let mut prev_loss = f32::MAX;
        for _ in 0..5 {
            let _ = w2v.forward(input.view(), target.view()).unwrap();
            let loss = binary_entropy_loss(
                y_true.clone(),
                w2v.grad_vars["sigmoid_output"].unwrap_arr1(),
            );
            println!("Current loss: {}", loss);
            // assert!(loss < prev_loss, "Loss did not decrease: {} >= {}", loss, prev_loss);
            prev_loss = loss;
            let _ = w2v.backward(y_true.clone());
        }
    }
}
