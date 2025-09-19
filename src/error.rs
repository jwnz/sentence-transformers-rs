use thiserror::Error;

#[derive(Debug, Error)]
pub enum SentenceTransformerBuilderError {
    #[error("Device must be specified")]
    DeviceNotSpecified,

    #[error("Pooling method must be specified")]
    PoolingMethodNotSpecified,

    #[error("DownloadHFModelError: {0}")]
    DownloadHFModelError(#[from] DownloadHFModelError),

    #[error("IO Error: {0}")]
    StdIOError(#[from] std::io::Error),

    #[error("SerdeJsonError: {0}")]
    SerdeJsonError(#[from] serde_json::Error),

    #[error("LoadSafeTensorError: {0}")]
    LoadSafeTensorError(#[from] LoadSafeTensorError),

    #[error("CandleError: {0}")]
    CandleError(#[from] candle_core::error::Error),

    #[error("TokenizersError: {0}")]
    TokenizersError(#[from] tokenizers::Error),

    #[error("DenseError: {0}")]
    DenseError(#[from] DenseError),
}

#[derive(Debug, Error)]
pub enum DownloadHFModelError {
    #[error("HFHubApiError({0})")]
    HFHubApiError(#[from] hf_hub::api::sync::ApiError),
}

#[derive(Debug, Error)]
pub enum FastTokenBatchError {
    #[error("CandleError({0})")]
    CandleError(#[from] candle_core::error::Error),
}

#[derive(Debug, Error)]
pub enum LoadSafeTensorError {
    #[error("CandleError({0})")]
    CandleError(#[from] candle_core::error::Error),
}

#[derive(Debug, Error)]
pub enum PoolingError {
    #[error("CandleError({0})")]
    CandleError(#[from] candle_core::error::Error),
}

#[derive(Debug, Error)]
pub enum DenseError {
    #[error("CandleError({0})")]
    CandleError(#[from] candle_core::error::Error),

    #[error("ActivationError({0})")]
    ActivationError(#[from] ActivationError),
}

#[derive(Debug, Error)]
pub enum ActivationError {
    #[error("CandleError({0})")]
    CandleError(#[from] candle_core::error::Error),
}

#[derive(Error, Debug)]
pub enum EmbedError {
    #[error("TokenizersError({0})")]
    TokenizersError(#[from] tokenizers::Error),

    #[error("CandleError({0})")]
    CandleError(#[from] candle_core::error::Error),

    #[error("PoolingError({0})")]
    PoolingError(#[from] PoolingError),

    #[error("FastTokenBatchError({0})")]
    FastTokenBatchError(#[from] FastTokenBatchError),

    #[error("DenseError({0})")]
    DenseError(#[from] DenseError),

    #[error("NormalizeError({0})")]
    NormalizeError(#[from] NormalizeError),
}

#[derive(Debug, Error)]
pub enum NormalizeError {
    #[error("CandleError({0})")]
    CandleError(#[from] candle_core::error::Error),
}

#[derive(Error, Debug)]
pub enum CosineSimilarityError {
    #[error("Cosine similarity of 0 sized vectors is undefined")]
    ZeroSizedVectorSimUndefined,

    #[error("Cosine similarity of vectors of different lengths (lhs: {lhs:?}, rhs: {rhs:?}) is undefined")]
    DifferentLenVectorSimUndefined { lhs: usize, rhs: usize },
}

#[derive(Error, Debug)]
pub enum LoadConfigError {
    #[error("IO Error: {0}")]
    StdIOError(#[from] std::io::Error),

    #[error("SerdeJsonError: {0}")]
    SerdeJsonError(#[from] serde_json::Error),
}
