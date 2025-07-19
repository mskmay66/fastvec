use pyo3::prelude::*;
use rayon::prelude::*;
use crate::vocab::Vocab;
use random_word::Lang;
use itertools::Itertools;

fn negative_sample(input: usize, vocabulary: &Vocab, num_samples: usize) -> Vec<(usize, usize, u8)> {
    let mut samples = Vec::new();
    for _ in 0..num_samples {
        let random_index = vocabulary.get_random_id(Some(input)).unwrap();
        samples.push((input, random_index, 0)); // Negative sample
    }
    samples
}

#[pyclass]
pub enum Example {
    W2V(usize, usize, u8),
    D2V(usize, usize, usize, u8),
}

#[pyclass]
pub struct Builder {
    #[pyo3(get)]
    documents: Vec<Vec<String>>,
    vocab: Vocab,
    window: Option<usize>,
}

#[pymethods]
impl Builder {
    #[new]
    pub fn new(documents: Vec<Vec<String>>, vocab: Vocab, window: Option<usize>) -> Self {
        Builder {
            documents,
            vocab,
            window,
        }
    }

    pub fn build_example(&self, encoded_doc: Vec<usize>, context_window: usize, doc_index: Option<usize>) ->  Vec<Example> {
        let mut examples = Vec::new();
        for w in encoded_doc.windows(context_window) {
            for permutation in w.iter().cartesian_product(w.iter()) {
                let (word, context_word) = (permutation.0, permutation.1);
                if word == context_word {
                    continue; // Skip if the word is the same as the context word
                }

                if let Some(doc_index) = doc_index {
                    examples.push(Example::D2V(doc_index, word.clone(), context_word.clone(), 1));
                    examples.extend(negative_sample(word.clone(), &self.vocab, 5).into_iter().map(|(input, sample, label)| Example::D2V(doc_index, input, sample, label)));
                } else {
                    examples.push(Example::W2V(word.clone(), context_word.clone(), 1));
                    examples.extend(negative_sample(word.clone(), &self.vocab, 5).into_iter().map(|(input, sample, label)| Example::W2V(input, sample, label)));
                }
            }
        }
        examples
    }

    pub fn build_w2v_training(&self) -> PyResult<Vec<Example>> {
        let context_window: usize = self.window.unwrap_or(5);
        let examples = self.documents.par_iter().map(|doc| {
            let encoded_doc: Vec<usize> = self.vocab.get_ids(doc.clone()).unwrap_or_else(|_| vec![]);
            self.build_example(encoded_doc, context_window, None)
        }).flatten().collect::<Vec<_>>();
        Ok(examples)
    }

    pub fn build_d2v_training(&self) -> PyResult<Vec<Example>> {
        let context_window: usize = self.window.unwrap_or(5);
        let examples = self.documents.par_iter().enumerate().map(|(i, doc)| {
            let encoded_doc: Vec<usize> = self.vocab.get_ids(doc.clone()).unwrap_or_else(|_| vec![]);
            self.build_example(encoded_doc, context_window, Some(i))
        }).flatten().collect::<Vec<_>>();
        Ok(examples)
    }
}

fn generate_random_documents(num_docs: usize, num_words: usize) -> Vec<Vec<String>> {
    (0..num_docs)
        .map(|_| {
            (0..num_words)
                .map(|_| random_word::get(Lang::En).to_string())
                .collect()
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocab;

    #[test]
    fn test_builder_creation() {
        let documents = vec![vec!["word1".to_string(), "word2".to_string()], vec!["word3".to_string()]];
        let vocab = Vocab::from_words(documents.iter().flat_map(|doc| doc.clone()).collect());
        let builder = Builder::new(documents, vocab, Some(5));
        assert_eq!(builder.documents.len(), 2);
        assert_eq!(builder.vocab.size, 3);
        assert_eq!(builder.window, Some(5));
    }

    #[test]
    fn test_build_example() {
        let documents = vec![vec!["word1".to_string(), "word2".to_string(), "word3".to_string()]];
        let vocab = Vocab::from_words(documents.iter().flat_map(|doc| doc.clone()).collect());
        let builder = Builder::new(documents, vocab, Some(3));

        let encoded_doc: Vec<usize> = builder.vocab.get_ids(vec!["word1".to_string(), "word2".to_string(), "word3".to_string()]).unwrap();
        let examples = builder.build_example(encoded_doc.clone(), 3, None);

        assert!(!examples.is_empty());
        assert!(examples.iter().all(|e| matches!(e, Example::W2V(_, _, _))));

        let d2v_examples = builder.build_example(encoded_doc, 3, Some(0));
        assert!(!d2v_examples.is_empty());
        assert!(d2v_examples.iter().all(|e| matches!(e, Example::D2V(_, _, _, _))));
    }

    #[test]
    fn test_build_w2v_training() {
        let documents = generate_random_documents(25, 10);
        let vocab = Vocab::from_words(documents.iter().flat_map(|doc| doc.clone()).collect());
        assert!(!vocab.word_to_id.is_empty());
        assert!(!vocab.words.is_empty());
        assert!(vocab.size > 0);
        assert!(!vocab.valid_ids.is_empty());

        let builder = Builder::new(documents, vocab, Some(3));

        let examples = builder.build_w2v_training().unwrap();
        assert!(!examples.is_empty());
        assert!(examples.iter().all(|e| matches!(e, Example::W2V(_, _, _))));
    }

    #[test]
    fn test_build_d2v_training() {
        let documents = generate_random_documents(25, 10);
        let vocab = Vocab::from_words(documents.iter().flat_map(|doc| doc.clone()).collect());
        assert!(!vocab.word_to_id.is_empty());
        assert!(!vocab.words.is_empty());
        assert!(vocab.size > 0);
        assert!(!vocab.valid_ids.is_empty());
        let builder = Builder::new(documents, vocab, Some(3));

        let examples = builder.build_d2v_training().unwrap();
        assert!(!examples.is_empty());
        assert!(examples.iter().all(|e| matches!(e, Example::D2V(_, _, _, _))));
    }

    #[test]
    fn test_negative_sample() {
        let mut vocab = Vocab::from_words(vec!["word1".to_string(), "word2".to_string(), "word3".to_string(), "word4".to_string(), "word5".to_string()]);
        vocab.size = 5; // Assuming we have 5 words in the vocabulary
        vocab.valid_ids = vec![0, 1, 2, 3, 4]; // Assuming word1=0, word2=1, word3=2
        let samples = negative_sample(0, &vocab, 2);
        assert_eq!(samples.len(), 2);
        for (input, sample, label) in samples {
            assert_eq!(input, 0);
            assert!(vocab.valid_ids.contains(&sample));
            assert_eq!(label, 0);
        }
    }
}
