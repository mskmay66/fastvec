use pyo3::prelude::*;

mod vocab;
mod embedding;
mod build;
mod preprocessing;
mod word2vec;
mod doc2vec;

use vocab::Vocab;
use embedding::Embedding;
use build::Builder;
use build::TrainingSet;
use preprocessing::simple_preprocessing;
use word2vec::{W2V, binary_entropy_loss};
// use doc2vec::DocumentLayer;
use rayon::prelude::*;
use ndarray::{Array2, s};


#[pyfunction]
fn train_word2vec(training_set: TrainingSet, embedding_dim: usize, lr: f32, epochs: usize) -> PyResult<Embedding> {
    let mut w2v = W2V::new(embedding_dim, lr);
    let mut embeddings = Embedding::new(embedding_dim);
    for epoch in 0..epochs {
        for (input, target, label) in &training_set {
            let input_array: Array2<f32> = Array2::from_shape_vec((1, input.len()), input.clone()).unwrap();
            let target_array: Array2<f32> = Array2::from_shape_vec((1, target.len()), target.clone()).unwrap();
            let label_array: Array2<u32> = Array2::from_shape_vec((1, label.len()), label.clone()).unwrap();

            // Forward pass
            let (out, input_output, target_output) = w2v.forward(input_array.view(), target_array.view()).unwrap();

            // Calculate loss
            let loss = binary_entropy_loss(label_array, out.clone());

            // Backward pass
            w2v.backward(loss, input_array, target_array, out.clone(), input_output, target_output);

            if epoch == epochs - 1 {
                embeddings.add_vectors(
                    input.iter().map(|&i| i as usize).collect(),
                    out.axis_iter(ndarray::Axis(0)).map(|row| row.to_vec()).collect()
                )?;
            }
        }
    }
    Ok(embeddings)
}


#[pymodule]
fn fastvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vocab>()?;
    m.add_class::<Embedding>()?;
    m.add_class::<Builder>()?;
    m.add_class::<TrainingSet>()?;
    m.add_class::<preprocessing::Tokens>()?;
    m.add_function(wrap_pyfunction!(simple_preprocessing, m)?)?;
    m.add_function(wrap_pyfunction!(train_word2vec, m)?)?;
    // m.add_class::<_Word2Vec>()?;
    // m.add_class::<DocumentLayer>()?;
    Ok(())
}
