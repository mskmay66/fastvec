use pyo3::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;
use crate::vocab::Vocab;

fn negative_sample(input: usize, vocabulary: &Vocab, num_samples: usize) -> Vec<(usize, usize, u8)> {
    let mut samples = Vec::new();
    let mut rng = rand::rng();
    for _ in 0..num_samples {
        let random_index = rng.random_range(0..(vocabulary.size));
        if (random_index != input) && vocabulary.valid_ids.contains(&random_index) {
            samples.push((input, random_index, 0)); // Negative sample
        }
    }
    samples
}

#[pyclass]
pub enum Example {
    W2V(usize, usize, u8),
    D2V(usize, usize, usize, u8),
    // D2VInference(usize, usize, u8),
}

#[pyclass]
pub struct Builder {
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
            let center = w[context_window / 2];
            for (i, &word) in w.iter().enumerate() {
                if i != context_window / 2 {
                    if let Some(doc_index) = doc_index {
                        examples.push(Example::D2V(doc_index, center, word, 1));
                        examples.extend(negative_sample(center, &self.vocab, 5).into_iter().map(|(input, sample, label)| Example::D2V(doc_index, input, sample, label)));
                    } else {
                        examples.push(Example::W2V(center, word, 1));
                        examples.extend(negative_sample(center, &self.vocab, 5).into_iter().map(|(input, sample, label)| Example::W2V(input, sample, label)));
                    }
                }
            }
        }
        examples
    }


    // pub fn build_d2v_inference(&self, docs: Vec<Vec<String>>, num_pairs: usize) -> PyResult<Vec<Example>> {
    //     let context_window: usize = self.window.unwrap_or(5);
    //     let mut examples = Vec::new();
    //     let mut rng = rand::thread_rng();
    //     let examples = docs.par_iter().map(|doc| {
    //         let words: Vec<String> = doc.choose_multiple(&mut rng, num_pairs).collect();
    //         let encoded_words: Vec<usize> = self.vocab.get_ids(words).unwrap_or_else(|_| vec![]);
    //         for word in encoded_words {
    //             examples.push(Example::D2VInference(word, 0, 1)); // Assuming 0 as a placeholder for the second parameter
    //         }
    //     }).collect::<Vec<_>>();
    //     for doc in docs.par_iter() {
    //         let words: Vec<String> = doc.choose_multiple(&mut rng, num_pairs).collect();

    //         // if let Some(random_word) = doc.choose_multiple(&mut rng, num_pairs) {
    //         //     // let encoded_doc: Vec<usize> = self.vocab.get_ids(doc.to_vec()).unwrap_or_else(|_| vec![]);
    //         //     let encoded_doc: Vec<usize> = self.vocab.get_ids(random_word.to_vec()).unwrap_or_else(|_| vec![]);
    //         //     example.extend()
    //         // }
    //         // let encoded_doc: Vec<usize> = self.vocab.get_ids(doc.to_vec()).unwrap_or_else(|_| vec![]);
    //         // examples.extend(self.build_example(encoded_doc, context_window, None));
    //     }
    //     Ok(examples)
    // }

    pub fn build_w2v_training(&self) -> PyResult<Vec<Example>> {
        let context_window: usize = self.window.unwrap_or(5);
        let examples = self.documents.par_iter().map(|doc| {
            let encoded_doc: Vec<usize> = self.vocab.get_ids(doc.to_vec()).unwrap_or_else(|_| vec![]);
            self.build_example(encoded_doc, context_window, None)
        }).flatten().collect::<Vec<_>>();
        Ok(examples)
    }

    pub fn build_d2v_training(&self) -> PyResult<Vec<Example>> {
        let context_window: usize = self.window.unwrap_or(5);
        let examples = self.documents.par_iter().enumerate().map(|(i, doc)| {
            let encoded_doc: Vec<usize> = self.vocab.get_ids(doc.to_vec()).unwrap_or_else(|_| vec![]);
            self.build_example(encoded_doc, context_window, Some(i))
        }).flatten().collect::<Vec<_>>();
        Ok(examples)
    }
}
