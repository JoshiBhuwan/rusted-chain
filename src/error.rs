//! Errors that bubble up through the Python bindings.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RustedChainError {
    #[error("API error {status}: {message}")]
    Api { status: u16, message: String },

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("Tool '{0}' not found in tools_dict")]
    ToolNotFound(String),

    #[error("Max iterations ({0}) reached without getting a final answer")]
    MaxIterations(usize),

    #[error("Tool '{0}' was requested but invoke() only supports tool schemas. Use run_with_tools(query, tools_dict) to provide executable tool functions.")]
    ToolExecutionNotSupported(String),

    #[error("No valid response from API")]
    NoResponse,
}

impl RustedChainError {
    pub fn api_error(status: reqwest::StatusCode, message: String) -> Self {
        Self::Api {
            status: status.as_u16(),
            message,
        }
    }
}

impl From<RustedChainError> for pyo3::PyErr {
    fn from(err: RustedChainError) -> pyo3::PyErr {
        match &err {
            RustedChainError::ToolNotFound(_) => {
                pyo3::PyErr::new::<pyo3::exceptions::PyKeyError, _>(err.to_string())
            }
            RustedChainError::ToolExecutionNotSupported(_) => {
                pyo3::PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(err.to_string())
            }
            _ => pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()),
        }
    }
}

// Keep accepting plain strings from older call sites.
impl From<String> for RustedChainError {
    fn from(s: String) -> Self {
        RustedChainError::ParseError(s)
    }
}

