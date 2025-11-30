mod claude;
mod error;
mod gemini;
mod openai;

use claude::{Claude, ContentBlock as ClaudeContentBlock, Message as ClaudeMessage};
use serde_json::json;
use dotenv;
#[allow(unused_imports)]
use error::RustedChainError;
use gemini::{Gemini, GeminiResponse, Content as GeminiContent, Part as GeminiPart, FunctionResponseData};
use once_cell::sync::Lazy;
use openai::{OpenAI, Message as OpenAIMessage};
use pyo3::prelude::*;
use tokio::runtime::Runtime;

const MAX_TOOL_ITERATIONS: usize = 10;

static RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("Failed to create tokio runtime"));

enum Provider {
    Gemini,
    OpenAI,
    Claude,
}

fn detect_provider(model: &str) -> PyResult<Provider> {
    const OPENAI_MODELS: &[&str] = &[
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o1-mini",
        "o1-preview",
        "o3-mini",
    ];
    const CLAUDE_MODELS: &[&str] = &[
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "claude-3.5-sonnet",
        "claude-sonnet-4-5",
    ];
    const GEMINI_MODELS: &[&str] = &[
        "gemini-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    ];

    for openai_model in OPENAI_MODELS {
        if model.starts_with(openai_model) {
            return Ok(Provider::OpenAI);
        }
    }

    for claude_model in CLAUDE_MODELS {
        if model.starts_with(claude_model) {
            return Ok(Provider::Claude);
        }
    }

    for gemini_model in GEMINI_MODELS {
        if model.starts_with(gemini_model) {
            return Ok(Provider::Gemini);
        }
    }

    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "Unknown model '{}'. Supported models:\n  OpenAI: {}\n  Claude: {}\n  Gemini: {}",
        model,
        OPENAI_MODELS.join(", "),
        CLAUDE_MODELS.join(", "),
        GEMINI_MODELS.join(", ")
    )))
}

fn convert_tools(py: Python, tools: &Option<Vec<Py<PyAny>>>) -> Vec<serde_json::Value> {
    tools
        .as_ref()
        .map(|t| {
            t.iter()
                .map(|tool| {
                    let tool_bound = tool.bind(py);
                    // Prefer the wrapper-provided schema if it exists.
                    if let Ok(schema) = tool_bound.call_method0("to_dict") {
                        pythonize::depythonize(&schema).unwrap_or(serde_json::Value::Null)
                    } else {
                        // Otherwise treat whatever we received as plain dict data.
                        pythonize::depythonize(tool_bound).unwrap_or(serde_json::Value::Null)
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn wrap_tool_result(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(_) => value,
        other => json!({ "result": other }),
    }
}

#[pyfunction]
#[pyo3(signature = (model, tools=None, api_key=None))]
fn create_agent(
    py: Python,
    model: String,
    tools: Option<Vec<Py<PyAny>>>,
    api_key: Option<String>,
) -> PyResult<Py<PyAny>> {
    dotenv::dotenv().ok();

    let provider = detect_provider(&model)?;

    match provider {
        Provider::OpenAI => {
            let agent = OpenAIModel {
                model: Some(model),
                tools,
                api_key,
            };
            Ok(Py::new(py, agent)?.into())
        }
        Provider::Claude => {
            let agent = ClaudeModel {
                model: Some(model),
                tools,
                api_key,
            };
            Ok(Py::new(py, agent)?.into())
        }
        Provider::Gemini => {
            let agent = GeminiModel {
                model: Some(model),
                tools,
                api_key,
            };
            Ok(Py::new(py, agent)?.into())
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ToolCall {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub args: String,
}

#[pymethods]
impl ToolCall {
    fn __repr__(&self) -> String {
        format!("ToolCall(name='{}', args={})", self.name, self.args)
    }
}

#[pyclass]
pub enum AgentResponse {
    Text { text: String },
    ToolCall { tool_call: ToolCall },
}

#[pymethods]
impl AgentResponse {
    #[getter]
    fn is_text(&self) -> bool {
        matches!(self, AgentResponse::Text { .. })
    }

    #[getter]
    fn is_tool_call(&self) -> bool {
        matches!(self, AgentResponse::ToolCall { .. })
    }

    #[getter]
    fn text(&self) -> PyResult<String> {
        match self {
            AgentResponse::Text { text } => Ok(text.clone()),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Response is not a text response",
            )),
        }
    }

    #[getter]
    fn tool_call(&self) -> PyResult<ToolCall> {
        match self {
            AgentResponse::ToolCall { tool_call } => Ok(tool_call.clone()),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Response is not a tool call",
            )),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            AgentResponse::Text { text } => format!("AgentResponse.Text('{}')", text),
            AgentResponse::ToolCall { tool_call } => {
                format!("AgentResponse.ToolCall({})", tool_call.__repr__())
            }
        }
    }
}

#[pyclass]
pub struct GeminiModel {
    model: Option<String>,
    tools: Option<Vec<Py<PyAny>>>,
    api_key: Option<String>,
}

impl GeminiModel {
    /// Build a configured Gemini client (internal method)
    fn build_client(&self, py: Python) -> Gemini {
        let mut client = Gemini::new();
        if let Some(m) = &self.model {
            client = client.with_model(m.clone());
        }
        if let Some(k) = &self.api_key {
            client = client.with_api_key(k.clone());
        }
        let tools_json = convert_tools(py, &self.tools);
        if !tools_json.is_empty() {
            client = client.with_tools(tools_json);
        }
        client
    }
}

#[pymethods]
impl GeminiModel {
    #[new]
    #[pyo3(signature = (model=None, tools=None, api_key=None))]
    fn new(
        model: Option<String>,
        tools: Option<Vec<Py<PyAny>>>,
        api_key: Option<String>,
    ) -> Self {
        GeminiModel {
            model,
            tools,
            api_key,
        }
    }

    fn add_tool(&mut self, tool: Py<PyAny>) {
        if let Some(tools) = &mut self.tools {
            tools.push(tool);
        } else {
            self.tools = Some(vec![tool]);
        }
    }

    /// Invoke the model and return the response (text or tool call).
    /// Like LangChain's invoke() - single shot, doesn't auto-execute tools.
    fn invoke(&self, py: Python, query: String) -> PyResult<AgentResponse> {
        let client = self.build_client(py);

        let response = RUNTIME.block_on(async {
            client
                .invoke_with_response(&query)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        })?;

        match response {
            GeminiResponse::Text(text) => Ok(AgentResponse::Text { text }),
            GeminiResponse::ToolCall(tool_call) => Ok(AgentResponse::ToolCall {
                tool_call: ToolCall {
                    name: tool_call.name,
                    args: serde_json::to_string(&tool_call.args)
                        .unwrap_or_else(|_| "{}".to_string()),
                },
            }),
        }
    }

    /// Run the model with automatic tool execution.
    /// Like LangChain's AgentExecutor - loops until final text response.
    fn run(&self, py: Python, query: String) -> PyResult<String> {
        let tools_dict = pyo3::types::PyDict::new(py);
        let mut has_tools = false;
        if let Some(tools) = &self.tools {
            for tool in tools {
                let tool_obj = tool.bind(py);
                if let Ok(name) = tool_obj.getattr("__name__") {
                    tools_dict.set_item(name, tool_obj)?;
                    has_tools = true;
                }
            }
        }

        if !has_tools {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "run() requires at least one callable tool. Use invoke() for tool-free calls.",
            ));
        }

        let client = self.build_client(py);
        let mut conversation = vec![GeminiContent {
            parts: vec![GeminiPart::Text {
                text: query.clone(),
            }],
            role: Some("user".to_string()),
        }];

        for _iteration in 0..MAX_TOOL_ITERATIONS {
            let (response, assistant_content) = RUNTIME.block_on(async {
                client
                    .exchange(conversation.clone())
                    .await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            })?;

            conversation.push(assistant_content);

            match response {
                GeminiResponse::Text(text) => {
                    return Ok(text);
                }
                GeminiResponse::ToolCall(tool_call) => {
                    let tool_fn = tools_dict
                        .get_item(&tool_call.name)?
                        .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                            "Tool '{}' not found",
                            tool_call.name
                        ))
                        })?;

                    let kwargs = pythonize::pythonize(py, &tool_call.args)?;
                    let result = if let Ok(dict) = kwargs.cast::<pyo3::types::PyDict>() {
                        tool_fn.call((), Some(&dict))?
                    } else {
                        tool_fn.call0()?
                    };

                    let result_value =
                        pythonize::depythonize(&result).unwrap_or(serde_json::Value::Null);
                    let response_json = wrap_tool_result(result_value);

                    conversation.push(GeminiContent {
                        parts: vec![GeminiPart::FunctionResponse {
                            function_response: FunctionResponseData {
                                name: tool_call.name.clone(),
                                response: response_json,
                            },
                        }],
                        role: Some("function".to_string()),
                    });
                }
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Max iterations reached without getting a final answer",
        ))
    }
}

#[pyclass]
pub struct OpenAIModel {
    model: Option<String>,
    tools: Option<Vec<Py<PyAny>>>,
    api_key: Option<String>,
}

impl OpenAIModel {
    /// Build a configured OpenAI client (internal method)
    fn build_client(&self, py: Python) -> OpenAI {
        let mut client = OpenAI::new();
        if let Some(m) = &self.model {
            client = client.with_model(m.clone());
        }
        if let Some(k) = &self.api_key {
            client = client.with_api_key(k.clone());
        }
        let tools_json = convert_tools(py, &self.tools);
        if !tools_json.is_empty() {
            client = client.with_tools(tools_json);
        }
        client
    }
}

#[pymethods]
impl OpenAIModel {
    #[new]
    #[pyo3(signature = (model=None, tools=None, api_key=None))]
    fn new(
        model: Option<String>,
        tools: Option<Vec<Py<PyAny>>>,
        api_key: Option<String>,
    ) -> Self {
        OpenAIModel {
            model,
            tools,
            api_key,
        }
    }

    fn add_tool(&mut self, tool: Py<PyAny>) {
        if let Some(tools) = &mut self.tools {
            tools.push(tool);
        } else {
            self.tools = Some(vec![tool]);
        }
    }

    /// Invoke the model and return the response (text or tool call).
    /// Like LangChain's invoke() - single shot, doesn't auto-execute tools.
    fn invoke(&self, py: Python, query: String) -> PyResult<AgentResponse> {
        let client = self.build_client(py);

        let response = RUNTIME.block_on(async {
            client
                .invoke_with_response(&query)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        })?;

        match response {
            openai::OpenAIResponse::Text(text) => Ok(AgentResponse::Text { text }),
            openai::OpenAIResponse::ToolCall(tool_call) => Ok(AgentResponse::ToolCall {
                tool_call: ToolCall {
                    name: tool_call.name,
                    args: serde_json::to_string(&tool_call.args)
                        .unwrap_or_else(|_| "{}".to_string()),
                },
            }),
        }
    }

    /// Run the model with automatic tool execution.
    /// Like LangChain's AgentExecutor - loops until final text response.
    fn run(&self, py: Python, query: String) -> PyResult<String> {
        let tools_dict = pyo3::types::PyDict::new(py);
        let mut has_tools = false;
        if let Some(tools) = &self.tools {
            for tool in tools {
                let tool_obj = tool.bind(py);
                if let Ok(name) = tool_obj.getattr("__name__") {
                    tools_dict.set_item(name, tool_obj)?;
                    has_tools = true;
                }
            }
        }

        if !has_tools {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "run() requires at least one callable tool. Use invoke() for tool-free calls.",
            ));
        }

        let client = self.build_client(py);
        let mut conversation = vec![OpenAIMessage {
            role: "user".to_string(),
            content: query.clone(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }];

        for _iteration in 0..MAX_TOOL_ITERATIONS {
            let (response, assistant_message) = RUNTIME.block_on(async {
                client
                    .chat(conversation.clone())
                    .await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            })?;

            conversation.push(assistant_message);

            match response {
                openai::OpenAIResponse::Text(text) => {
                    return Ok(text);
                }
                openai::OpenAIResponse::ToolCall(tool_call) => {
                    let tool_fn = tools_dict
                        .get_item(&tool_call.name)?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                                "Tool '{}' not found",
                                tool_call.name
                            ))
                        })?;

                    let kwargs = pythonize::pythonize(py, &tool_call.args)?;
                    let result = if let Ok(dict) = kwargs.cast::<pyo3::types::PyDict>() {
                        tool_fn.call((), Some(&dict))?
                    } else {
                        tool_fn.call0()?
                    };

                    let result_value =
                        pythonize::depythonize(&result).unwrap_or(serde_json::Value::Null);
                    let result_text =
                        serde_json::to_string(&result_value).unwrap_or_else(|_| "null".to_string());

                    conversation.push(OpenAIMessage {
                        role: "tool".to_string(),
                        content: result_text,
                        name: None,
                        tool_call_id: Some(tool_call.id.clone()),
                        tool_calls: None,
                    });
                }
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Max iterations reached without getting a final answer",
        ))
    }
}

#[pyclass]
pub struct ClaudeModel {
    model: Option<String>,
    tools: Option<Vec<Py<PyAny>>>,
    api_key: Option<String>,
}

impl ClaudeModel {
    /// Build a configured Claude client (internal method)
    fn build_client(&self, py: Python) -> Claude {
        let mut client = Claude::new();
        if let Some(m) = &self.model {
            client = client.with_model(m.clone());
        }
        if let Some(k) = &self.api_key {
            client = client.with_api_key(k.clone());
        }
        let tools_json = convert_tools(py, &self.tools);
        if !tools_json.is_empty() {
            client = client.with_tools(tools_json);
        }
        client
    }
}

#[pymethods]
impl ClaudeModel {
    #[new]
    #[pyo3(signature = (model=None, tools=None, api_key=None))]
    fn new(
        model: Option<String>,
        tools: Option<Vec<Py<PyAny>>>,
        api_key: Option<String>,
    ) -> Self {
        ClaudeModel {
            model,
            tools,
            api_key,
        }
    }

    fn add_tool(&mut self, tool: Py<PyAny>) {
        if let Some(tools) = &mut self.tools {
            tools.push(tool);
        } else {
            self.tools = Some(vec![tool]);
        }
    }

    /// Invoke the model and return the response (text or tool call).
    /// Like LangChain's invoke() - single shot, doesn't auto-execute tools.
    fn invoke(&self, py: Python, query: String) -> PyResult<AgentResponse> {
        let client = self.build_client(py);

        let response = RUNTIME.block_on(async {
            client
                .invoke_with_response(&query)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        })?;

        match response {
            claude::ClaudeResponse::Text(text) => Ok(AgentResponse::Text { text }),
            claude::ClaudeResponse::ToolCall(tool_call) => Ok(AgentResponse::ToolCall {
                tool_call: ToolCall {
                    name: tool_call.name,
                    args: serde_json::to_string(&tool_call.args)
                        .unwrap_or_else(|_| "{}".to_string()),
                },
            }),
        }
    }

    /// Run the model with automatic tool execution.
    /// Like LangChain's AgentExecutor - loops until final text response.
    fn run(&self, py: Python, query: String) -> PyResult<String> {
        let tools_dict = pyo3::types::PyDict::new(py);
        let mut has_tools = false;
        if let Some(tools) = &self.tools {
            for tool in tools {
                let tool_obj = tool.bind(py);
                if let Ok(name) = tool_obj.getattr("__name__") {
                    tools_dict.set_item(name, tool_obj)?;
                    has_tools = true;
                }
            }
        }

        if !has_tools {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "run() requires at least one callable tool. Use invoke() for tool-free calls.",
            ));
        }

        let client = self.build_client(py);
        let mut conversation = vec![ClaudeMessage {
            role: "user".to_string(),
            content: vec![ClaudeContentBlock::Text {
                text: query.clone(),
            }],
        }];

        for _iteration in 0..MAX_TOOL_ITERATIONS {
            let (response, assistant_message) = RUNTIME.block_on(async {
                client
                    .exchange(conversation.clone())
                    .await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            })?;

            conversation.push(assistant_message);

            match response {
                claude::ClaudeResponse::Text(text) => {
                    return Ok(text);
                }
                claude::ClaudeResponse::ToolCall(tool_call) => {
                    let tool_fn = tools_dict
                        .get_item(&tool_call.name)?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                                "Tool '{}' not found",
                                tool_call.name
                            ))
                        })?;

                    let kwargs = pythonize::pythonize(py, &tool_call.args)?;
                    let result = if let Ok(dict) = kwargs.cast::<pyo3::types::PyDict>() {
                        tool_fn.call((), Some(&dict))?
                    } else {
                        tool_fn.call0()?
                    };

                    let result_value =
                        pythonize::depythonize(&result).unwrap_or(serde_json::Value::Null);
                    let wrapped_result = wrap_tool_result(result_value);

                    conversation.push(ClaudeMessage {
                        role: "user".to_string(),
                        content: vec![ClaudeContentBlock::ToolResult {
                            tool_use_id: tool_call.id.clone(),
                            content: wrapped_result,
                        }],
                    });
                }
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Max iterations reached without getting a final answer",
        ))
    }
}

#[pymodule]
fn rusted_chain(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_agent, m)?)?;
    m.add_class::<GeminiModel>()?;
    m.add_class::<OpenAIModel>()?;
    m.add_class::<ClaudeModel>()?;
    m.add_class::<AgentResponse>()?;
    m.add_class::<ToolCall>()?;
    Ok(())
}
