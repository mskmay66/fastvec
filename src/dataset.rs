use rayon::prelude::*;


pub struct DataSet {
    pub documents: Vec<Vec<String>>,
    pub vocab: Option<Vocab>,
    pub context_window: usize,
}

impl DataSet {
    pub fn new(documents: Vec<Vec<String>>) -> Self {
        DataSet {
            documents,
            context_window: 5, // Default context window size
            vocab: None,
        }
    }

    pub fn set_vocab(&mut self, vocab: Vocab) {
        self.vocab = Some(vocab);
    }

    pub fn get_vocab(&self) -> Option<&Vocab> {
        self.vocab.as_ref()
    }

    pub fn get_documents(&self) -> &Vec<Vec<String>> {
        &self.documents
    }

    pub fn add_document(&mut self, document: Vec<String>) {
        self.documents.push(document);
        for word in &document {
            if let Some(vocab) = &mut self.vocab {
                vocab.add_word(word.clone());
            }
        }
    }

    pub fn from_documents(documents: Vec<Vec<String>>, context_window: Option<usize>) -> Self {
        let mut dataset = DataSet::new(documents, context_window.unwrap_or(5));
        let vocab = Vocab::from_words(
            dataset.documents.par_iter().flat_map(|doc| doc.iter().cloned()).collect()
        );
        dataset.set_vocab(vocab);
        dataset
    }

    fn negative_sample(&self, pos: usize, num: usize) -> Vec<(usize, usize, u8)> {
        let mut neg = Vec::new();
        for 0..num {
            let mut rng = rand::thread_rng();
            let random_index = rng.gen_range(0..self.vocab.as_ref().unwrap().size);
            let random_word = &self.vocab.as_ref().unwrap().valid_ids[random_index];
            if random_word != pos {
                neg.push((pos, *random_word, 0));
            }
        }
        neg
    }

    pub fn build(&self) -> Vec<(usize, usize, u8)> {
        let mut examples = Vec::new();
        &self.documents.par_iter().for_each(|doc| {
            let encoded_doc: Vec<usize> = self.vocab.as_ref().unwrap().get_ids(doc.clone());
            for window in encoded_doc.windows(self.context_window) {
                let center = window[self.context_window / 2];
                for (i, &word) in window.iter().enumerate() {
                    if i != self.context_window / 2 {
                        examples.push((center, word, 1));
                        examples.extend(self.negative_sample(center, 5));
                    }
                }
            }
        });
        examples
    }

}