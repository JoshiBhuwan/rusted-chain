use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;

#[derive(Serialize, Clone)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
}

#[derive(Serialize, Clone)]
struct Tool {
    function_declarations: Vec<Value>,
}

#[derive(Serialize, Clone)]
pub(crate) struct Content {
    pub(crate) parts: Vec<Part>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
}

#[derive(Serialize, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum Part {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCallData,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponseData,
    },
}

#[derive(Serialize, Clone, Deserialize)]
pub(crate) struct FunctionCallData {
    pub(crate) name: String,
    pub(crate) args: Value,
}

#[derive(Serialize, Clone, Deserialize)]
pub(crate) struct FunctionResponseData {
    pub(crate) name: String,
    pub(crate) response: Value,
}

#[derive(Deserialize)]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Deserialize)]
struct Candidate {
    content: ContentResponse,
}

#[derive(Deserialize)]
struct ContentResponse {
    parts: Vec<Part>,
    #[allow(dead_code)]
    role: Option<String>,
}

pub struct Gemini {
    api_key: String,
    model: String,
    client: Client,
    tools: Option<Vec<serde_json::Value>>,
}

impl Default for Gemini {
    fn default() -> Self {
        dotenv::dotenv().ok();
        Self {
            api_key: env::var("GOOGLE_API_KEY").unwrap_or_default(),
            model: "gemini-2.5-flash".to_string(),
            client: Client::new(),
            tools: None,
        }
    }
}

#[derive(Clone)]
pub struct ToolCall {
    pub name: String,
    pub args: Value,
}

pub enum GeminiResponse {
    Text(String),
    ToolCall(ToolCall),
}

impl Gemini {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = api_key;
        self
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    pub fn with_tools(mut self, tools: Vec<serde_json::Value>) -> Self {
        self.tools = Some(tools);
        self
    }

    async fn send_request(
        &self,
        contents: Vec<Content>,
    ) -> Result<GenerateContentResponse, String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let tools = self.tools.as_ref().map(|t| {
            vec![Tool {
                function_declarations: t.clone(),
            }]
        });

        let request_body = GenerateContentRequest { contents, tools };

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Failed to send request: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("API Error {}: {}", status, text));
        }

        response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))
    }

    #[allow(dead_code)]
    pub async fn invoke(&self, prompt: &str) -> Result<String, String> {
        let contents = vec![Content {
            parts: vec![Part::Text {
                text: prompt.to_string(),
            }],
            role: Some("user".to_string()),
        }];

        const MAX_ITERATIONS: usize = 10;
        for _ in 0..MAX_ITERATIONS {
            let response = self.send_request(contents.clone()).await?;

            if let Some(candidates) = response.candidates {
                if let Some(candidate) = candidates.first() {
                    let parts = &candidate.content.parts;

                    for part in parts {
                        if let Part::Text { text } = part {
                            return Ok(text.clone());
                        }
                    }

                    return Err("Function calling not yet supported in invoke()".to_string());
                }
            }
        }

        Err("Max iterations reached without getting a text response".to_string())
    }

    pub async fn invoke_with_response(&self, prompt: &str) -> Result<GeminiResponse, String> {
        let contents = vec![Content {
            parts: vec![Part::Text {
                text: prompt.to_string(),
            }],
            role: Some("user".to_string()),
        }];

        let response = self.send_request(contents).await?;

        if let Some(candidates) = response.candidates {
            if let Some(candidate) = candidates.first() {
                let parts = &candidate.content.parts;

                for part in parts {
                    match part {
                        Part::Text { text } => {
                            return Ok(GeminiResponse::Text(text.clone()));
                        }
                        Part::FunctionCall { function_call } => {
                            return Ok(GeminiResponse::ToolCall(ToolCall {
                                name: function_call.name.clone(),
                                args: function_call.args.clone(),
                            }));
                        }
                        _ => {}
                    }
                }
            }
        }

        Err("No valid response from Gemini".to_string())
    }

    #[allow(dead_code)]
    #[allow(private_interfaces)]
    pub async fn continue_with_tool_result(
        &self,
        conversation_history: Vec<Content>,
        tool_call: &ToolCall,
        result: Value,
    ) -> Result<String, String> {
        let mut contents = conversation_history;

        contents.push(Content {
            parts: vec![Part::FunctionResponse {
                function_response: FunctionResponseData {
                    name: tool_call.name.clone(),
                    response: result,
                },
            }],
            role: Some("function".to_string()),
        });

        let response = self.send_request(contents).await?;

        if let Some(candidates) = response.candidates {
            if let Some(candidate) = candidates.first() {
                for part in &candidate.content.parts {
                    if let Part::Text { text } = part {
                        return Ok(text.clone());
                    }
                }
            }
        }

        Err("No text response after tool execution".to_string())
    }

    pub(crate) async fn exchange(
        &self,
        conversation: Vec<Content>,
    ) -> Result<(GeminiResponse, Content), String> {
        let response = self.send_request(conversation.clone()).await?;

        if let Some(candidates) = response.candidates {
            if let Some(candidate) = candidates.first() {
                let assistant_content = Content {
                    parts: candidate.content.parts.clone(),
                    role: candidate.content.role.clone(),
                };

                for part in &candidate.content.parts {
                    match part {
                        Part::Text { text } => {
                            return Ok((GeminiResponse::Text(text.clone()), assistant_content));
                        }
                        Part::FunctionCall { function_call } => {
                            return Ok((
                                GeminiResponse::ToolCall(ToolCall {
                                    name: function_call.name.clone(),
                                    args: function_call.args.clone(),
                                }),
                                assistant_content,
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }

        Err("No valid response from Gemini".to_string())
    }
}
