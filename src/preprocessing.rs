use unicode_normalization::char::decompose_canonical;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct Tokens {
    #[pyo3(get)]
    tokens: Vec<Vec<String>>
}

#[pymethods]
impl Tokens {
    #[new]
    pub fn new(tokens: Vec<Vec<String>>) -> Self {
        Tokens { tokens }
    }

    pub fn flatten(&self) -> PyResult<Vec<String>> {
        let t = self.tokens.clone();
        Ok(t.into_iter().flatten().collect())
    }
}

fn deaccent_char(c: char) -> char {
    let mut base_char = None;
    decompose_canonical(c, |c| { base_char.get_or_insert(c); });
    base_char.unwrap_or(c)
}

fn deaccent(text: &str) -> String {
    text.chars()
        .map(deaccent_char)
        .collect()
}

fn tokenize(text: &str, deacc: Option<bool>) -> Vec<String> {
    let _deacc = deacc.unwrap_or(false);
    text.split_whitespace()
        .map(|s| { 
            if _deacc {
                deaccent(s).to_lowercase()
            } else {
                s.to_lowercase()
            }
        })
        .collect()
}

#[pyfunction]
pub fn simple_preprocessing(corpus: Vec<String>, deacc: Option<bool>) -> PyResult<Tokens> {
    Ok(Tokens::new(corpus.par_iter()
        .map(|doc| tokenize(&doc, deacc))
        .collect()))
}