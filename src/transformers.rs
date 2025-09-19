use tokenizers::Tokenizer;

use crate::models::{bert::BertModel, distilbert::DistilBertModel, TransformerLoad};

pub struct Transformer<M: TransformerLoad> {
    model: M,
    tokenizer: Tokenizer,
}

impl Transformer<BertModel> {
    fn forward() {}
}

impl Transformer<DistilBertModel> {
    fn asdf() {}
}
