use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use unicode_normalization::char::canonical_combining_class;
use unicode_normalization::UnicodeNormalization;

pub struct Tokenizer {
    deacc: bool,
    lowercase: bool,
    split_pattern: Option<Regex>,
}

impl Tokenizer {
    pub fn new() -> Self {
        Tokenizer {
            deacc: false,
            lowercase: true,
            split_pattern: None,
        }
    }

    pub fn deaccent(mut self, yes: bool) -> Self {
        self.deacc = yes;
        self
    }
    pub fn lowercase(mut self, yes: bool) -> Self {
        self.lowercase = yes;
        self
    }
    pub fn split_regex(mut self, pat: &str) -> Self {
        self.split_pattern = Some(Regex::new(pat).expect("Invalid regex pattern"));
        self
    }

    pub fn tokenize(
        &self,
        text: &str,
        min_len: Option<usize>,
        max_len: Option<usize>,
    ) -> Vec<String> {
        let min_len = min_len.unwrap_or(2);
        let max_len = max_len.unwrap_or(15);
        let PAT_ALPHABETIC = Regex::new(r"\p{Letter}+").expect("Invalid regex pattern");
        let raw: Vec<String> = if let Some(ref re) = self.split_pattern {
            re.split(text)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        } else {
            text.split_whitespace().map(|s| s.to_string()).collect()
        };

        raw.into_iter()
            .map(|mut s| {
                if self.deacc {
                    s = deaccent(&s);
                }
                if self.lowercase {
                    s = s.to_lowercase();
                }
                s
            })
            .filter(|s| {
                if s.is_empty() {
                    false
                } else {
                    s.len() >= min_len && s.len() <= max_len && PAT_ALPHABETIC.is_match(&s)
                }
            })
            .collect()
    }
}

fn deaccent(text: &str) -> String {
    text.nfd()
        .filter(|c| canonical_combining_class(*c) == 0)
        .collect()
}

#[pyclass]
pub struct Tokens {
    #[pyo3(get)]
    tokens: Vec<Vec<String>>,
}

#[pymethods]
impl Tokens {
    #[new]
    pub fn new(tokens: Vec<Vec<String>>) -> Self {
        Tokens { tokens }
    }

    pub fn flatten(&self) -> PyResult<Vec<String>> {
        Ok(self
            .tokens
            .iter()
            .flat_map(|doc| doc.iter().cloned())
            .collect())
    }

    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.tokens.len())
    }

    pub fn __getitem__(&self, idx: isize) -> PyResult<Vec<String>> {
        let len = self.tokens.len() as isize;
        let i = if idx < 0 { len + idx } else { idx } as usize;
        if i >= self.tokens.len() {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of range",
            ))
        } else {
            Ok(self.tokens[i].clone())
        }
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("<Tokens [{} documents]>", self.tokens.len()))
    }

    pub fn __iter__(slf: PyRefMut<Self>) -> PyResult<Py<Tokens>> {
        Ok(slf.into())
    }

    pub fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<Vec<String>>> {
        Ok(if slf.tokens.is_empty() {
            None
        } else {
            Some(slf.tokens.remove(0))
        })
    }
}

#[pyfunction]
#[pyo3(signature = (corpus, deacc=None, lowercase=None, split_pattern=None))]
pub fn simple_preprocessing(
    corpus: Vec<String>,
    deacc: Option<bool>,
    lowercase: Option<bool>,
    split_pattern: Option<String>,
) -> PyResult<Tokens> {
    let mut tok = Tokenizer::new();
    if let Some(d) = deacc {
        tok = tok.deaccent(d);
    }
    if let Some(l) = lowercase {
        tok = tok.lowercase(l);
    }
    if let Some(p) = split_pattern {
        tok = tok.split_regex(&p);
    }
    let tokens = corpus
        .par_iter()
        .map(|doc| tok.tokenize(doc, Some(2), Some(15)))
        .collect();
    Ok(Tokens::new(tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deaccent() {
        assert_eq!(deaccent("é"), "e");
        assert_eq!(deaccent("ÀçÇ"), "AcC");
    }

    #[test]
    fn test_tokenizer_defaults() {
        let t = Tokenizer::new();
        assert_eq!(
            t.tokenize("Hello WORLD!", Some(2), Some(15)),
            vec!["hello", "world!"]
        );
    }

    #[test]
    fn test_tokenizer_deaccent_lowercase() {
        let t = Tokenizer::new().deaccent(true).lowercase(true);
        assert_eq!(
            t.tokenize("Éxâmple Déjà Vu", Some(2), Some(15)),
            vec!["example", "deja", "vu"]
        );
    }

    #[test]
    fn test_alphabets_only() {
        let t = Tokenizer::new().deaccent(true).lowercase(true);
        assert_eq!(
            t.tokenize("Éxâmple Déjà Vu 123", Some(2), Some(15)),
            vec!["example", "deja", "vu"]
        );
    }

    #[test]
    fn test_tokenizer_split_regex() {
        let t = Tokenizer::new().split_regex(r"\W+");
        assert_eq!(
            t.tokenize("Hello, world!", Some(2), Some(15)),
            vec!["hello", "world"]
        );
    }

    #[test]
    fn test_flatten() {
        let toks = Tokens::new(vec![vec!["a".into(), "b".into()], vec!["c".into()]]);
        assert_eq!(toks.flatten().unwrap(), vec!["a", "b", "c"]);
    }
}
