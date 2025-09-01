extern crate blas_src;
extern crate ndarray;
use crossbeam::scope;
use crossbeam_channel::unbounded;
use ndarray::{s, Array1, ArrayView1, ArrayView2};
use std::cmp;
use std::sync::{Arc, Mutex};

use crate::embedding::Embedding;
use crate::utils::{array_to_vec, binary_entropy_grad, sigmoid, Layer};

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

    pub fn train(
        &mut self,
        input: ArrayView2<f32>,
        context: ArrayView2<f32>,
        y_true: ArrayView1<f32>,
        num_workers: usize,
        batch_size: usize,
    ) -> Result<(), String> {
        if (input.len() != context.len()) | (context.len() != y_true.len()) {
            return Err(String::from("All arrays must be the same length"));
        }

        let n = input.len();
        let working_index = Arc::new(Mutex::new(batch_size));
        let input_layer = Arc::new(Mutex::new(self.input_layer.clone()));
        let context_layer = Arc::new(Mutex::new(self.context_layer.clone()));
        scope(|s| {
            let mut handles = vec![];
            for _ in 0..num_workers {
                let handle = s.spawn(|_| {
                    loop {
                        let mut idx = working_index.lock().unwrap();
                        let _idx = *idx; // copy working index
                        if _idx == n {
                            break; // if the array has been consumed
                        }

                        let next = cmp::min(_idx + batch_size, n);
                        *idx = next;
                        drop(idx); // release the lock

                        // slice data
                        let input_batch = input.slice(s![_idx..next, ..]);
                        let context_batch = context.slice(s![_idx..next, ..]);
                        let y_true_batch = y_true.slice(s![_idx..next]);

                        // acqure input lock + forward pass
                        let thread_input_layer = input_layer.lock().unwrap();
                        let input_embedding = thread_input_layer.forward(input_batch);

                        // acquire context lock + forward pass
                        let thread_context_layer = context_layer.lock().unwrap();
                        let context_embedding = thread_context_layer.forward(input_batch);

                        // release the locks
                        drop(thread_input_layer);
                        drop(thread_context_layer);

                        let consine_sim = (0..input_embedding.shape()[0])
                            .map(|i| {
                                let input_vec = input_embedding.row(i);
                                let context_vec = context_embedding.row(i);
                                input_vec.dot(&context_vec)
                            })
                            .collect::<Array1<f32>>();

                        let sig = sigmoid(consine_sim);

                        // Backward pass
                        let loss: f32 = binary_entropy_grad(y_true_batch, sig);
                        let context_embedding_grad = loss * input_embedding; // gradient w.r.t. context word embedding
                        let input_embedding_grad = loss * context_embedding; // gradient w.r.t. input word embedding
                        let input_grad = input_batch.t().dot(&input_embedding_grad); // gradient w.r.t. input layer weights
                        let target_grad = context_batch.t().dot(&context_embedding_grad); // gradient w.r.t. target layer weights

                        // apply gradients
                        // acquire weights lock
                        {
                            let mut thread_input_layer = input_layer.lock().unwrap();
                            thread_input_layer.weights -= &(input_grad * self.lr);
                        }
                        {
                            let mut thread_context_layer = context_layer.lock().unwrap();
                            thread_context_layer.weights -= &(target_grad * self.lr);
                        }
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        })
        .unwrap();
        Ok(())
    }

    pub fn predict(
        &mut self,
        input: ArrayView2<f32>,
        batch_size: usize,
        num_workers: usize,
    ) -> Result<Embedding, String> {
        let mut embeddings = Embedding::new(self.embedding_dim);

        // Clone weights once to avoid locking in threads
        let weights = self.input_layer.weights.clone();

        let n = input.len();
        let working_index = Arc::new(Mutex::new(0)); // start at 0

        // Crossbeam thread scope
        scope(|s| {
            let (tx, rx) = unbounded();

            // Spawn worker threads
            for _ in 0..num_workers {
                let thread_tx = tx.clone();
                let thread_weights = weights.clone();
                let thread_input = input.to_owned(); // owned copy for thread
                let thread_idx = Arc::clone(&working_index);

                s.spawn(move |_| {
                    loop {
                        // Lock and get current index
                        let mut idx = thread_idx.lock().unwrap();
                        if *idx >= n {
                            break;
                        }
                        let next = cmp::min(*idx + batch_size, n);
                        let slice_idx = *idx;
                        *idx = next; // advance for next iteration
                        drop(idx);

                        // Slice owned input batch
                        let input_batch = thread_input.slice(s![slice_idx..next, ..]).to_owned();

                        // Compute embeddings
                        let res = input_batch.dot(&thread_weights);

                        // Send batch result
                        thread_tx.send((input_batch, res)).unwrap();
                    }
                });
            }

            // Drop original sender so receiver knows when all threads are done
            drop(tx);

            // Collect results from threads
            for (input_batch, embedding) in rx {
                let _ = embeddings.add_vectors(
                    array_to_vec(input_batch)
                        .into_iter()
                        .flatten()
                        .map(|x| x as usize)
                        .collect(),
                    array_to_vec(embedding),
                );
            }
        })
        .unwrap();
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::binary_entropy_loss;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_w2v_creation() {
        let w2v = W2V::new(10, 0.01);
        assert_eq!(w2v.embedding_dim, 10);
        assert_eq!(w2v.lr, 0.01);
        assert_eq!(w2v.input_layer.weights.shape(), &[1, 10]);
        assert_eq!(w2v.context_layer.weights.shape(), &[1, 10]);
    }

    #[test]
    fn test_train() {
        let mut w2v = W2V::new(5, 0.01);
        let input = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let context = Array2::from_shape_vec((3, 1), vec![4.0, 5.0, 6.0]).unwrap();
        let y_true = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let result = w2v.train(input.view(), context.view(), y_true.view(), 1, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_predict() {
        let mut w2v = W2V::new(5, 0.01);

        let input = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let context = Array2::from_shape_vec((3, 1), vec![4.0, 5.0, 6.0]).unwrap();
        let y_true = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let _ = w2v
            .train(input.view(), context.view(), y_true.view(), 3, 1)
            .unwrap();
        let prediction = w2v.predict(input.view(), 3, 1).unwrap();
        assert_eq!(prediction.vectors.len(), 3);
    }
}
