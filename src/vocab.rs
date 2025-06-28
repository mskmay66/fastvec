use rayon::prelude::*;

pub struct Vocab {
    pub size: usize,
    pub words: Vec<String>,
    pub word_to_id: std::collections::HashMap<String, usize>,
}

impl Vocab {
    pub fn new() -> Self {
        Vocab {
            size: 0,
            words: Vec::new(),
            word_to_id: std::collections::HashMap::new(),
        }
    }

    pub fn from_words(words: Vec<String>) -> Self {
        let mut vocab = Vocab::new();
        words.par_iter().for_each(|word| {
            vocab.add_word(word);
        });
        vocab
    }

    pub fn add_word(&mut self, word: String) {
        if !self.word_to_id.contains_key(&word) {
            let id = self.size;
            self.word_to_id.insert(word.clone(), id);
            self.words.push(word);
            self.size += 1;
        }
    }

    pub fn get_id(&self, word: &str) -> Option<usize> {
        self.word_to_id.get(word).cloned()
    }
}