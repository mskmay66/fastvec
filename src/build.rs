use pyo3::prelude::*;
use rayon::prelude::*;
use crate::vocab::Vocab;
use random_word::Lang;
use itertools::Itertools;

fn negative_sample(window: Vec<usize>, input: usize, vocabulary: &Vocab, num_samples: usize) -> Vec<(usize, usize, u8)> {
    let mut samples = Vec::new();
    let w = Some(window);
    for _ in 0..num_samples {
        let random_index = vocabulary.get_random_id(w.clone()).unwrap();
        samples.push((input, random_index, 0)); // Negative sample
    }
    samples
}


#[pyclass]
#[derive(Clone)]
pub struct TrainingSet {
    #[pyo3(get)]
    pub input_words: Vec<usize>,
    #[pyo3(get)]
    pub context_words: Vec<usize>,
    #[pyo3(get)]
    pub labels: Vec<u8>,
    #[pyo3(get)]
    pub batch_size: usize,
    curr: usize,
    next: usize,
}

#[pymethods]
impl TrainingSet {
    #[new]
    pub fn new(input_words: Vec<usize>, context_words: Vec<usize>, labels: Vec<u8>, batch_size: Option<usize>) -> Self {
        let curr = 0;
        let next = 0;
        TrainingSet {
            input_words,
            context_words,
            labels,
            batch_size: batch_size.unwrap_or(32), // Default batch size
            curr,
            next
        }
    }

    pub fn add_example(&mut self, input: usize, context: usize, label: u8) {
        self.input_words.push(input);
        self.context_words.push(context);
        self.labels.push(label);
    }

    pub fn extend(&mut self, other: TrainingSet) -> PyResult<()> {
        self.input_words.extend(other.input_words);
        self.context_words.extend(other.context_words);
        self.labels.extend(other.labels);
        Ok(())
    }

    pub fn get_batch(&self, start: usize, end: usize) -> PyResult<(Vec<usize>, Vec<usize>, Vec<u8>)> {
        if start >= self.input_words.len() || end > self.input_words.len() || start >= end {
            return Err(pyo3::exceptions::PyIndexError::new_err("Invalid range for batch"));
        }
        let input_batch = self.input_words[start..end].to_vec();
        let context_batch = self.context_words[start..end].to_vec();
        let labels_batch = self.labels[start..end].to_vec();
        Ok((input_batch, context_batch, labels_batch))
    }

    pub fn __len__(&self) -> usize {
        self.input_words.len()
    }
}

impl Iterator for TrainingSet {
    type Item = (Vec<usize>, Vec<usize>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr >= self.input_words.len() {
            return None;
        }
        if self.curr + self.batch_size > self.input_words.len() {
            self.next = self.input_words.len();
        } else {
            self.next = self.curr + self.batch_size;
        }

        let (input, context, labels) = self.get_batch(self.curr, self.next).unwrap();
        self.curr = self.next;
        Some((input, context, labels))
    }
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

    fn build_example(&self, encoded_doc: Vec<usize>, context_window: usize) -> PyResult<TrainingSet> {
        let mut training_set = TrainingSet::new(Vec::new(), Vec::new(), Vec::new(), None);
        for w in encoded_doc.windows(context_window) {
            for permutation in w.iter().cartesian_product(w.iter()) {
                let (word, context_word) = (permutation.0, permutation.1);
                if word == context_word {
                    continue; // Skip if the word is the same as the context word
                }
                training_set.add_example(word.clone(), context_word.clone(), 1);
                negative_sample(w.to_vec(), word.clone(), &self.vocab, 5).into_iter().for_each(|(input, sample, label)| {
                    training_set.add_example(input, sample, label);
                });
            }
        }
        Ok(training_set)
    }

    pub fn build_training(&self, batch_size: Option<usize>) -> PyResult<TrainingSet> {
        let context_window: usize = self.window.unwrap_or(5);
        let mut training_set = TrainingSet::new(Vec::new(), Vec::new(), Vec::new(), batch_size);
        self.documents.iter().for_each(|doc| {
            let encoded_doc = self.vocab.get_ids(doc.clone()).unwrap();
            training_set.extend(self.build_example(encoded_doc, context_window).expect("Failed to build example"));
        });
        Ok(training_set)
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
        let training_set = builder.build_example(encoded_doc.clone(), 3).unwrap();

        assert!(training_set.input_words.len() > 0);
        assert!(training_set.context_words.len() > 0);
        assert!(training_set.labels.len() > 0);
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

        let training_set = builder.build_training(Some(32)).unwrap();
        assert!(training_set.input_words.len() > 0);
        assert!(training_set.context_words.len() > 0);
        assert!(training_set.labels.len() > 0);

        // Check that the training set can be iterated over
        let mut iter = training_set.into_iter();
        assert!(iter.next().is_some());
    }

    #[test]
    fn test_negative_sample() {
        let mut vocab = Vocab::from_words(vec!["word1".to_string(), "word2".to_string(), "word3".to_string(), "word4".to_string(), "word5".to_string()]);
        vocab.size = 5; // Assuming we have 5 words in the vocabulary
        vocab.valid_ids = vec![0, 1, 2, 3, 4]; // Assuming word1=0, word2=1, word3=2
        let samples = negative_sample(vec![0, 1], 0, &vocab, 2);
        assert_eq!(samples.len(), 2);
        for (input, sample, label) in samples {
            assert_eq!(input, 0);
            assert!(vocab.valid_ids.contains(&sample));
            assert_eq!(label, 0);
        }
    }
}
