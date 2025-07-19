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
        Ok(words.par_iter().filter_map(|word| self.get_id(word).unwrap()).collect())
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
        match self.word_to_id.get(word) {
            Some(&id) => Ok(Some(id)),
            None => Ok(None),
        }
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_creation() {
        let empty_vocab = Vocab::new();
        assert_eq!(empty_vocab.size, 0);
        assert!(empty_vocab.words.is_empty());
        assert!(empty_vocab.word_to_id.is_empty());
        assert!(empty_vocab.valid_ids.is_empty());

        let words = vec!["apple".to_string(), "banana".to_string(), "apple".to_string()];
        let vocab = Vocab::from_words(words);
        assert_eq!(vocab.size, 2);
        assert_eq!(vocab.words, vec!["apple", "banana"]);
        assert_eq!(vocab.word_to_id.get("apple"), Some(&0));
        assert_eq!(vocab.word_to_id.get("banana"), Some(&1));
    }

    #[test]
    fn test_add_word() {
        let mut vocab = Vocab::new();
        assert!(vocab.add_word("test").is_ok());
        assert_eq!(vocab.size, 1);
        assert_eq!(vocab.words, vec!["test"]);
        assert_eq!(vocab.word_to_id.get("test"), Some(&0));

        // Adding the same word again should not change the size
        assert!(vocab.add_word("test").is_ok());
        assert_eq!(vocab.size, 1);
    }

    #[test]
    fn test_get_id() {
        let mut vocab = Vocab::new();
        vocab.add_word("apple").unwrap();
        vocab.add_word("banana").unwrap();

        assert_eq!(vocab.get_id("apple").unwrap(), Some(0));
        assert_eq!(vocab.get_id("banana").unwrap(), Some(1));
        assert_eq!(vocab.get_id("cherry").unwrap(), None); // Cherry not added
    }

    #[test]
    fn test_get_ids() {
        let words = vec!["apple".to_string(), "banana".to_string(), "cherry".to_string()];
        let mut vocab = Vocab::new();
        for word in &words {
            vocab.add_word(word).unwrap();
        }
        let ids = vocab.get_ids(words).unwrap();
        assert_eq!(ids, vec![0, 1, 2]); // Assuming apple=0, banana=1, cherry=2
    }

    #[test]
    fn test_subsample() {
        let mut word_to_freq = HashMap::new();
        word_to_freq.insert(0, 0.01);
        word_to_freq.insert(1, 0.02);
        word_to_freq.insert(2, 0.03);

        let vocab_size = 3;

        assert!(subsample(&0, &word_to_freq, vocab_size).is_some());
        assert!(subsample(&1, &word_to_freq, vocab_size).is_some());
        assert!(subsample(&2, &word_to_freq, vocab_size).is_some());

        // Test with a frequency that should not be sampled
        word_to_freq.insert(3, 0.0001);
        assert!(subsample(&3, &word_to_freq, vocab_size).is_none());
    }
}
