use pyo3::prelude::*;
use crate::vocab::Vocab;
use random_word::Lang;
// use itertools::Itertools;
use itertools::iproduct;

fn negative_sample(window: Vec<usize>, input: usize, vocabulary: &Vocab, num_samples: usize) -> Vec<(usize, usize, u32)> {
    let mut samples = Vec::new();
    let w = Some(window);
    for _ in 0..num_samples {
        let random_index = vocabulary.get_random_id(w.clone()).unwrap();
        samples.push((input, random_index, 0 as u32)); // Negative sample
    }
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
    pub fn new(input_words: Vec<usize>, context_words: Vec<usize>, labels: Vec<u32>, batch_size: Option<usize>) -> Self {
        let curr = 0;
        let next = 0;
        TrainingSet {
            input_words: input_words.iter().map(|&x| x as f32).collect(),
            context_words: context_words.iter().map(|&x| x as f32).collect(),
            labels,
            batch_size: batch_size.unwrap_or(32), // Default batch size
            curr,
            next
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

    pub fn get_item(&self, idx: usize) -> PyResult<(f32, f32, u32)> {
        if idx >= self.input_words.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Index out of range"));
        }
        Ok((
            self.input_words[idx],
            self.context_words[idx],
            self.labels[idx],
        ))
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

    fn build_example(&self, encoded_doc: Vec<usize>, context_window: usize) -> PyResult<TrainingSet> {
        if encoded_doc.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Encoded document is empty"));
        } else if (context_window == 0) {
            return Err(pyo3::exceptions::PyValueError::new_err("Context window size must be greater than 0"));
        } else if (encoded_doc.len() < context_window) {
            // this will happen implictly in the loop but we can handle it here
            return Ok(TrainingSet::new(Vec::new(), Vec::new(), Vec::new(), None));
        }

        let mut training_set = TrainingSet::new(Vec::new(), Vec::new(), Vec::new(), None);
        for w in encoded_doc.windows(context_window) {
            let _ = iproduct!(w.iter(), w.iter())
                .filter(|(word, context_word)| word != context_word) // Skip if the word is the same as the context word
                .for_each(|(word, context_word)| {
                    let input = *word;
                    let context = *context_word;
                    training_set.add_example(input, context, 1);
                    negative_sample(w.to_vec(), input, &self.vocab, 5).into_iter().for_each(|(input, sample, label)| {
                        training_set.add_example(input, sample, label);
                    });
                });
        }
        Ok(training_set)
    }

    pub fn build_training(&self, batch_size: Option<usize>) -> PyResult<TrainingSet> {
        let context_window: usize = self.window.unwrap_or(5);
        let mut training_set = TrainingSet::new(Vec::new(), Vec::new(), Vec::new(), batch_size);
        self.documents.iter().for_each(|doc| {
            let encoded_doc = self.vocab.get_ids(doc.clone()).unwrap();
            let _ = training_set.extend(self.build_example(encoded_doc, context_window).expect("Failed to build example"));
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

        let encoded_doc: Vec<usize> = (0..10).map(|_| {builder.vocab.get_random_id(None).unwrap()}).collect();
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
