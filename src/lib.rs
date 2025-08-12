use pyo3::prelude::*;

mod builder;
mod doc2vec;
mod embedding;
mod preprocessing;
mod utils;
mod vocab;
pub mod word2vec;

use builder::Builder;
use builder::TrainingSet;
use doc2vec::DocumentLayer;
use embedding::Embedding;
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use preprocessing::simple_preprocessing;
use vocab::Vocab;
use word2vec::W2V;

#[pyfunction]
pub fn train_word2vec(
    training_set: TrainingSet,
    embedding_dim: usize,
    lr: f32,
    epochs: usize,
) -> PyResult<Embedding> {
    let mut w2v = W2V::new(embedding_dim, lr);
    let mut embeddings = Embedding::new(embedding_dim);
    for epoch in 0..epochs {
        for (input, target, label) in &training_set {
            let input_array: Array2<f32> =
                Array2::from_shape_vec((input.len(), 1), input.clone()).unwrap();
            let target_array: Array2<f32> =
                Array2::from_shape_vec((target.len(), 1), target.clone()).unwrap();
            let label_array: Array1<u32> = Array1::from_vec(label.clone());

            // Forward pass
            let input_embedding = w2v
                .forward(input_array.view(), target_array.view())
                .unwrap();

            // Backward pass
            let _ = w2v.backward(label_array);

            if epoch == epochs - 1 {
                embeddings.add_vectors(
                    input_array
                        .mapv(|x| x as usize)
                        .rows()
                        .into_iter()
                        .map(|row| row[0])
                        .collect(),
                    input_embedding
                        .rows()
                        .into_iter()
                        .map(|row| row.to_vec())
                        .collect(),
                )?;
            }
        }
    }
    Ok(embeddings)
}

#[pyfunction]
pub fn infer_doc_vectors(
    word_embeddings: Vec<Vec<f32>>,
    epochs: usize,
    lr: f32,
) -> PyResult<Vec<Vec<f32>>> {
    let num_samples = word_embeddings.len();
    let dim = word_embeddings[0].len();
    let mut doc_layer = DocumentLayer::new(dim, lr);
    let doc_embedding: Array2<f32> = Array2::random((num_samples, 1), Uniform::new(-1.0, 1.0));
    let mut out: Array2<f32> = Array2::zeros((num_samples, dim));
    let word_vectors: Array2<f32> = Array2::from_shape_vec(
        (num_samples, dim),
        word_embeddings.into_iter().flatten().collect(),
    )
    .unwrap();

    for _ in 0..epochs {
        out = doc_layer.forward(doc_embedding.view(), word_vectors.view());
        let _ = doc_layer.backward(Array1::ones(num_samples));
    }

    Ok(out.outer_iter().map(|row| row.to_vec()).collect())
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
    m.add_function(wrap_pyfunction!(infer_doc_vectors, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_word2vec() {
        let documents = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["fast".to_string(), "vector".to_string()],
        ];
        let vocab = Vocab::from_words(documents.iter().flat_map(|doc| doc.clone()).collect(), 5);
        let builder = Builder::new(documents, vocab, Some(2));
        let training_set = builder.build_training(5, None).unwrap();

        let embedding_dim = 3;
        let lr = 0.01;
        let epochs = 1;

        let embeddings = train_word2vec(training_set, embedding_dim, lr, epochs).unwrap();
        println!("Embeddings: {:?}", embeddings.vectors);
        assert_eq!(embeddings.dim, embedding_dim);
        assert!(!embeddings.vectors.is_empty());
        assert!(embeddings.vectors.len() > 0);
    }

    #[test]
    fn test_infer_doc_vectors() {
        let word_embeddings = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
            vec![0.1, 0.3, 0.5],
        ];
        let epochs = 2;
        let lr = 0.01;

        let doc_vectors = infer_doc_vectors(word_embeddings, epochs, lr).unwrap();
        println!("Document Vectors: {:?}", doc_vectors);
        assert_eq!(doc_vectors.len(), 4);
        assert_eq!(doc_vectors[0].len(), 3);
    }
}
