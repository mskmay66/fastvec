use rayon::prelude::*;
use pyo3::prelude::*;
use std::collections::HashMap;


#[pyclass]
#[derive(Clone, Debug)]
pub struct Vocab {
    #[pyo3(get)]
    pub size: usize,
    #[pyo3(get)]
    pub words: Vec<String>,
    pub word_to_id: HashMap<String, usize>,
    pub valid_ids: Vec<usize>,
}

#[pymethods]
impl Vocab {
    #[new]
    pub fn new() -> Self {
        Vocab {
            size: 0,
            words: Vec::new(),
            word_to_id: HashMap::new(),
            valid_ids: Vec::new(),
        }
    }

    #[staticmethod]
    pub fn from_words(words: Vec<String>) -> Self {
        let mut vocab = Vocab::new();
        let mut word_to_freq: HashMap<usize, f64> = HashMap::new();
        words.iter().for_each(|word| {
            vocab.add_word(word);
            word_to_freq
                .entry(vocab.word_to_id[word])
                .and_modify(|e| *e += 1.0)
                .or_insert(1.0);
        });
        let n = vocab.size;
        vocab.valid_ids = words
            .par_iter()
            .filter_map(|word| { 
                if let Some(id) = vocab.word_to_id.get(word) {
                    subsample(id, &word_to_freq, n)
                } else {
                    None
                }
            })
            .collect();
        vocab
    }

    pub fn get_ids(&self, words: Vec<String>) -> PyResult<Vec<usize>> {
        Ok(words.par_iter().map(|word| self.word_to_id[word]).collect())
    }

    pub fn add_word(&mut self, word: &str) -> PyResult<()> {
        if !self.word_to_id.contains_key(word) {
            let id = self.size;
            self.word_to_id.insert(word.to_string(), id);
            self.words.push(word.to_string());
            self.size += 1;
            self.valid_ids.push(id);
        }
        Ok(())
    }

    pub fn get_id(&self, word: &str) -> PyResult<Option<usize>> {
        Ok(self.word_to_id.get(word).cloned())
    }
}

fn subsample(token_id: &usize, word_to_freq: &HashMap<usize, f64>, vocab_size: usize) -> Option<usize> {
        let freq = word_to_freq.get(token_id).unwrap_or(&0.1) / vocab_size as f64;
        let p = ((freq / 0.001).sqrt() + 1.0) * (0.001 / freq);
        let r: f64 = rand::random();
        if r < p {
            return Some(*token_id);
        }
        None
}