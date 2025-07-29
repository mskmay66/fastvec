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
use doc2vec::DocumentLayer;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;


#[pyfunction]
pub fn train_word2vec(training_set: TrainingSet, embedding_dim: usize, lr: f32, epochs: usize) -> PyResult<Embedding> {
    let mut w2v = W2V::new(embedding_dim, lr);
    let mut embeddings = Embedding::new(embedding_dim);
    for epoch in 0..epochs {
        for (input, target, label) in &training_set {
            let input_array: Array2<f32> = Array2::from_shape_vec((1, input.len()), input.clone()).unwrap();
            let target_array: Array2<f32> = Array2::from_shape_vec((1, target.len()), target.clone()).unwrap();
            let label_array: Array1<u32> = Array1::from_vec(label.clone());

            // Forward pass
            let input_embedding = w2v.forward(input_array.view(), target_array.view()).unwrap();

            // Backward pass
            let _ = w2v.backward(label_array);

            if epoch == epochs - 1 {
                embeddings.add_vectors(
                    input_array.mapv(|x| x as usize).rows().into_iter().map(|row| row[0]).collect(),
                    input_embedding.rows().into_iter().map(|row| row.to_vec()).collect(),
                )?;
            }
        }
    }
    Ok(embeddings)
}


// #[pyfunction]
// pub fn infer_doc_vectors(word_embeddings: Vec<Vec<f32>>, epochs: usize, lr: f32) -> PyResult<Vec<Vec<f32>>> {
//     let num_samples = word_embeddings.len();
//     let dim = word_embeddings[0].len();
//     let mut doc_layer = DocumentLayer::new(dim, lr);
//     let mut doc_vectors: Array2<f32> = Array2::random((num_samples, dim), Uniform::new(-1.0, 1.0));
//     let word_vectors: Array2<f32> = Array2::from_shape_vec((num_samples, dim), word_embeddings.into_iter().flatten().collect()).unwrap();
//     for _ in 0..epochs {
//         let (output, doc_vec) = doc_layer.forward(word_vectors.view(), doc_vectors.view());
//         doc_vectors = doc_vec;
//         doc_layer.backward(output.clone(), &doc_vectors, &word_vectors);
//     }
//     Ok(doc_vectors.axis_iter(ndarray::Axis(0)).map(|row| row.to_vec()).collect())
// }


#[pymodule]
fn fastvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vocab>()?;
    m.add_class::<Embedding>()?;
    m.add_class::<Builder>()?;
    m.add_class::<TrainingSet>()?;
    m.add_class::<preprocessing::Tokens>()?;
    m.add_function(wrap_pyfunction!(simple_preprocessing, m)?)?;
    m.add_function(wrap_pyfunction!(train_word2vec, m)?)?;
    // m.add_function(wrap_pyfunction!(infer_doc_vectors, m)?)?;
    Ok(())
}
