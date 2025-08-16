use pyo3::prelude::*;

pub mod builder;
mod data;
mod doc2vec;
mod embedding;
mod preprocessing;
mod utils;
mod vocab;
pub mod word2vec;

use builder::Builder;
use data::{DataLoader, Dataset};
use doc2vec::DocumentLayer;
use embedding::Embedding;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use preprocessing::simple_preprocessing;
use utils::array_to_vec;
use vocab::Vocab;
use word2vec::W2V;

#[pyfunction]
pub fn train_word2vec(
    training_set: Dataset,
    embedding_dim: usize,
    batch_size: Option<usize>,
    lr: f32,
    epochs: usize,
) -> PyResult<Embedding> {
    let mut w2v = W2V::new(embedding_dim, lr);
    let mut embeddings = Embedding::new(embedding_dim);
    let loader = DataLoader::from_dataset(&training_set, batch_size.unwrap_or(32));
    for epoch in 0..epochs {
        loader.iter().for_each(|(input, context, label)| {
            let _ = w2v.train_batch(
                input, context, label, // TODO rewrite grad function to accept view
            );

            if epoch == epochs - 1 {
                let embedding = w2v.predict(input).unwrap();
                embeddings.add_vectors(
                    array_to_vec(input.to_owned())
                        .into_iter()
                        .flatten() // flatten inner Vecs
                        .map(|x| x as usize) // cast f32 -> usize
                        .collect(),
                    array_to_vec(embedding),
                );
            }
        });
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
    let word_vectors: Array2<f32> = Array2::from_shape_vec(
        (num_samples, dim),
        word_embeddings.into_iter().flatten().collect(),
    )
    .unwrap();

    for _ in 0..epochs {
        let _ = doc_layer
            .train_batch(doc_embedding.view(), word_vectors.view())
            .unwrap();
    }

    let embeddings = doc_layer.predict(doc_embedding.view()).unwrap();

    Ok(array_to_vec(embeddings))
}

#[pymodule]
fn fastvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vocab>()?;
    m.add_class::<Embedding>()?;
    m.add_class::<Builder>()?;
    m.add_class::<Dataset>()?;
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
        let batch_size = Some(2);

        let embeddings =
            train_word2vec(training_set, embedding_dim, batch_size, lr, epochs).unwrap();
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
