//! Error types for lang_rain

use thiserror::Error;

/// Errors that can occur when interacting with LLM APIs
#[derive(Error, Debug)]
pub enum LangRainError {
    /// API returned an error response
    #[error("API error {status}: {message}")]
    Api { status: u16, message: String },

    /// Network/HTTP error
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Failed to parse API response
    #[error("Failed to parse response: {0}")]
    ParseError(String),

    /// Tool not found in tools dictionary
    #[error("Tool '{0}' not found in tools_dict")]
    ToolNotFound(String),

    /// Max iterations reached without getting a response
    #[error("Max iterations ({0}) reached without getting a final answer")]
    MaxIterations(usize),

    /// Function calling not supported in this context
    #[error("Tool '{0}' was requested but invoke() only supports tool schemas. Use run_with_tools(query, tools_dict) to provide executable tool functions.")]
    ToolExecutionNotSupported(String),

    /// No response generated
    #[error("No valid response from API")]
    NoResponse,
}

impl LangRainError {
    /// Create an API error from status code and response text
    pub fn api_error(status: reqwest::StatusCode, message: String) -> Self {
        Self::Api {
            status: status.as_u16(),
            message,
        }
    }
}

impl From<LangRainError> for pyo3::PyErr {
    fn from(err: LangRainError) -> pyo3::PyErr {
        match &err {
            LangRainError::ToolNotFound(_) => {
                pyo3::PyErr::new::<pyo3::exceptions::PyKeyError, _>(err.to_string())
            }
            LangRainError::ToolExecutionNotSupported(_) => {
                pyo3::PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(err.to_string())
            }
            _ => pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()),
        }
    }
}

// Helper to convert String errors to LangRainError (for backward compatibility)
impl From<String> for LangRainError {
    fn from(s: String) -> Self {
        LangRainError::ParseError(s)
    }
}

