use crate::vocab::Vocab;
use itertools::Itertools;
use pyo3::prelude::*;
use rayon::prelude::*;

fn negative_sample(
    window: Vec<usize>,
    input: usize,
    vocabulary: &Vocab,
    num_samples: usize,
) -> Vec<(usize, usize, u32)> {
    let samples = (0..num_samples)
        .into_par_iter()
        .map(|_| {
            let random_index = vocabulary.get_random_id(None).unwrap();
            (input, random_index, 0 as u32) // Negative sample
        })
        .collect();
    samples
}

#[pyclass]
#[derive(Clone)]
pub struct TrainingSet {
    #[pyo3(get)]
    pub input_words: Vec<f32>,
    #[pyo3(get)]
    pub context_words: Vec<f32>,
    #[pyo3(get)]
    pub labels: Vec<u32>,
    #[pyo3(get)]
    pub batch_size: usize,
    curr: usize,
    next: usize,
}

#[pymethods]
impl TrainingSet {
    #[new]
    pub fn new(
        input_words: Vec<usize>,
        context_words: Vec<usize>,
        labels: Vec<u32>,
        batch_size: Option<usize>,
    ) -> Self {
        let curr = 0;
        let next = 0;
        TrainingSet {
            input_words: input_words.par_iter().map(|&x| x as f32).collect(),
            context_words: context_words.par_iter().map(|&x| x as f32).collect(),
            labels,
            batch_size: batch_size.unwrap_or(32), // Default batch size
            curr,
            next,
        }
    }

    pub fn add_example(&mut self, input: usize, context: usize, label: u32) {
        self.input_words.push(input as f32);
        self.context_words.push(context as f32);
        self.labels.push(label);
    }

    pub fn extend(&mut self, other: TrainingSet) -> PyResult<()> {
        self.input_words.extend(other.input_words);
        self.context_words.extend(other.context_words);
        self.labels.extend(other.labels);
        Ok(())
    }

    pub fn get_batch(&self, start: usize, end: usize) -> PyResult<(Vec<f32>, Vec<f32>, Vec<u32>)> {
        if start >= self.input_words.len() || end > self.input_words.len() || start >= end {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Invalid range for batch",
            ));
        }
        let input_batch = self.input_words[start..end].to_vec();
        let context_batch = self.context_words[start..end].to_vec();
        let labels_batch = self.labels[start..end].to_vec();
        Ok((input_batch, context_batch, labels_batch))
    }

    pub fn __len__(&self) -> usize {
        self.input_words.len()
    }

    pub fn get_item(&self, idx: usize) -> PyResult<(f32, f32, u32)> {
        if idx >= self.input_words.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of range",
            ));
        }
        Ok((
            self.input_words[idx],
            self.context_words[idx],
            self.labels[idx],
        ))
    }

    pub fn is_empty(&self) -> bool {
        self.input_words.is_empty() && self.context_words.is_empty() && self.labels.is_empty()
    }
}

impl Iterator for TrainingSet {
    type Item = (Vec<f32>, Vec<f32>, Vec<u32>);

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

impl<'a> IntoIterator for &'a TrainingSet {
    type Item = (Vec<f32>, Vec<f32>, Vec<u32>);
    type IntoIter = TrainingSet;

    fn into_iter(self) -> Self::IntoIter {
        TrainingSet {
            input_words: self.input_words.clone(),
            context_words: self.context_words.clone(),
            labels: self.labels.clone(),
            batch_size: self.batch_size,
            curr: 0,
            next: 0,
        }
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

    fn build_example(
        &self,
        encoded_doc: Vec<usize>,
        num_neg: usize,
        context_window: usize,
    ) -> PyResult<(Vec<f32>, Vec<f32>, Vec<u32>)> {
        if encoded_doc.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Encoded document is empty",
            ));
        } else if context_window == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Context window size must be greater than 0",
            ));
        } else if encoded_doc.len() < context_window {
            // this will happen implicitly in the loop but we can handle it here
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        const COMBINATIONS_SIZE: usize = 2;
        let (input, context, label) = encoded_doc
            .windows(context_window)
            .flat_map(|w| {
                w.iter()
                    .combinations(COMBINATIONS_SIZE)
                    .filter(|pair| pair[0] != pair[1]) // Skip if the word is the same as the context word
                    .flat_map(|pair| {
                        let input = *pair[0];
                        let context = *pair[1];
                        let mut examples = vec![(input, context, 1)];
                        examples.extend(negative_sample(w.to_vec(), input, &self.vocab, num_neg));
                        examples
                    })
            })
            .fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |mut acc, (input, context, label)| {
                    acc.0.push(input as f32);
                    acc.1.push(context as f32);
                    acc.2.push(label);
                    acc
                },
            );
        Ok((input, context, label))
    }

    pub fn build_training(
        &self,
        num_neg: usize,
        batch_size: Option<usize>,
    ) -> PyResult<TrainingSet> {
        let context_window: usize = self.window.unwrap_or(5);
        if self.documents.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No documents available for training",
            ));
        }

        if self.vocab.size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Vocabulary is empty, cannot build training set",
            ));
        }

        if num_neg == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Number of negative samples must be greater than 0",
            ));
        }

        if context_window == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Context window size must be greater than 0",
            ));
        }

        if context_window > self.vocab.size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Context window size cannot be greater than vocabulary size",
            ));
        }

        let training_set = self
            .documents
            .par_iter()
            .map(|doc| {
                let encoded_doc = self.vocab.get_ids(doc.clone()).unwrap_or_else(|_| {
                    panic!("Failed to encode document: {:?}", doc);
                });
                self.build_example(encoded_doc, num_neg, context_window)
                    .expect("Failed to build example")
            })
            .fold(
                || (TrainingSet::new(Vec::new(), Vec::new(), Vec::new(), batch_size)),
                |mut acc, (input_words, context_words, labels)| {
                    acc.input_words.extend(input_words);
                    acc.context_words.extend(context_words);
                    acc.labels.extend(labels);
                    acc
                },
            )
            .reduce(
                || TrainingSet::new(Vec::new(), Vec::new(), Vec::new(), batch_size),
                |mut acc, training_set| {
                    acc.extend(training_set).unwrap();
                    acc
                },
            );

        if training_set.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No valid training examples were generated",
            ));
        }
        Ok(training_set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::Vocab;
    use random_word::Lang;

    fn generate_random_documents(num_docs: usize, num_words: usize) -> Vec<Vec<String>> {
        (0..num_docs)
            .map(|_| {
                (0..num_words)
                    .map(|_| random_word::get(Lang::En).to_string())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_builder_creation() {
        let documents = generate_random_documents(25, 10);
        let vocab = Vocab::from_words(documents.iter().flat_map(|doc| doc.clone()).collect());
        let builder = Builder::new(documents, vocab, Some(5));
        assert_eq!(builder.documents.len(), 25);
        assert!(builder.vocab.size > 0);
        assert!(builder.vocab.size <= 250);
        assert_eq!(builder.window, Some(5));
    }

    #[test]
    fn test_build_example() {
        let documents = generate_random_documents(25, 10);
        let vocab = Vocab::from_words(documents.iter().flat_map(|doc| doc.clone()).collect());
        let builder = Builder::new(documents, vocab, Some(5));

        let encoded_doc: Vec<usize> = (0..10)
            .map(|_| builder.vocab.get_random_id(None).unwrap())
            .collect();
        let training_set = builder.build_example(encoded_doc.clone(), 5, 3).unwrap();

        assert!(training_set.0.len() > 0);
        assert!(training_set.1.len() > 0);
        assert!(training_set.2.len() > 0);
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

        let training_set = builder.build_training(5, Some(32)).unwrap();
        assert!(training_set.input_words.len() > 0);
        assert!(training_set.context_words.len() > 0);
        assert!(training_set.labels.len() > 0);

        // Check that the training set can be iterated over
        let mut iter = training_set.into_iter();
        assert!(iter.next().is_some());
    }

    #[test]
    fn test_negative_sample() {
        let documents = generate_random_documents(25, 10);
        let vocab = Vocab::from_words(documents.iter().flat_map(|doc| doc.clone()).collect());
        let samples = negative_sample(vec![0, 1], 0, &vocab, 2);
        assert_eq!(samples.len(), 2);
        for (input, sample, label) in samples {
            assert_eq!(input, 0);
            assert!(vocab.valid_ids.contains(&sample));
            assert_eq!(label, 0);
        }
    }
}
