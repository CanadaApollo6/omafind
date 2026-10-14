//! Fast tier: Model2Vec static embeddings (token lookup + mean pooling), cheap enough
//! to embed every chunk on a whole disk with the CPU.

use anyhow::{Context, Result};
use model2vec_rs::model::StaticModel;

use crate::models;

pub struct StaticEmbedder {
    text: StaticModel,
    code: Option<StaticModel>,
    pub text_dim: usize,
    pub code_dim: usize,
}

/// Which static space a chunk lives in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Space {
    Text,
    Code,
}

const MAX_TOKENS: usize = 512;

impl StaticEmbedder {
    pub fn load(use_code_model: bool) -> Result<Self> {
        let dir = models::find_static(&models::STATIC_TEXT)?;
        let text = StaticModel::from_pretrained(&dir, None, Some(true), None)
            .with_context(|| format!("loading {}", dir.display()))?;
        let text_dim = text.encode_single("dim").len();
        let code = if use_code_model {
            let dir = models::find_static(&models::STATIC_CODE)?;
            Some(
                StaticModel::from_pretrained(&dir, None, Some(true), None)
                    .with_context(|| format!("loading {}", dir.display()))?,
            )
        } else {
            None
        };
        let code_dim = code.as_ref().map(|m| m.encode_single("dim").len()).unwrap_or(text_dim);
        Ok(Self { text, code, text_dim, code_dim })
    }

    fn model(&self, space: Space) -> &StaticModel {
        match (space, &self.code) {
            (Space::Code, Some(m)) => m,
            _ => &self.text,
        }
    }

    pub fn embed(&self, space: Space, texts: &[String]) -> Vec<Vec<f32>> {
        self.model(space).encode_with_args(texts, Some(MAX_TOKENS), 256)
    }

    pub fn embed_one(&self, space: Space, text: &str) -> Vec<f32> {
        self.embed(space, &[text.to_string()]).pop().unwrap_or_default()
    }
}
