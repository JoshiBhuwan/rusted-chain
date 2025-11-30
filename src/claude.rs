use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Serialize, Clone)]
pub(crate) struct Message {
    pub(crate) role: String,
    pub(crate) content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub(crate) enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(rename = "tool_use_id")]
        tool_use_id: String,
        content: serde_json::Value,
    },
}

pub enum ClaudeResponse {
    Text(String),
    ToolCall(ToolCall),
}

pub struct ToolCall {
    pub name: String,
    pub args: serde_json::Value,
    #[allow(dead_code)] // Reserved for multi-turn tool conversations
    pub id: String,
}

pub struct Claude {
    api_key: String,
    model: String,
    client: Client,
    tools: Option<Vec<serde_json::Value>>,
}

impl Default for Claude {
    fn default() -> Self {
        dotenv::dotenv().ok();
        Self {
            api_key: env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            model: "claude-sonnet-4-20250514".to_string(),
            client: Client::new(),
            tools: None,
        }
    }
}

impl Claude {
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

    #[allow(dead_code)]
    pub async fn invoke(&self, prompt: &str) -> Result<String, String> {
        match self.invoke_with_response(prompt).await? {
            ClaudeResponse::Text(text) => Ok(text),
            ClaudeResponse::ToolCall(tool_call) => {
                Ok(format!("Request to call tool: {}", tool_call.name))
            }
        }
    }

    pub async fn invoke_with_response(&self, prompt: &str) -> Result<ClaudeResponse, String> {
        let messages = vec![Message {
            role: "user".to_string(),
            content: vec![ContentBlock::Text {
                text: prompt.to_string(),
            }],
        }];

        let (response, _) = self.exchange(messages).await?;
        Ok(response)
    }

    pub(crate) async fn exchange(
        &self,
        messages: Vec<Message>,
    ) -> Result<(ClaudeResponse, Message), String> {
        let url = "https://api.anthropic.com/v1/messages";

        let request_body = MessagesRequest {
            model: self.model.clone(),
            max_tokens: 1024,
            messages,
            tools: self.tools.clone(),
        };

        let response = self
            .client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Failed to send request: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("API Error {}: {}", status, text));
        }

        let response_body: MessagesResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        let assistant_message = Message {
            role: "assistant".to_string(),
            content: response_body.content.clone(),
        };

        let mut text_response: Option<String> = None;
        for block in response_body.content {
            match block {
                ContentBlock::ToolUse { id, name, input } => {
                    return Ok((
                        ClaudeResponse::ToolCall(ToolCall {
                            name,
                            args: input,
                            id,
                        }),
                        assistant_message,
                    ));
                }
                ContentBlock::Text { text } => {
                    if text_response.is_none() {
                        text_response = Some(text);
                    }
                }
                _ => {}
            }
        }

        if let Some(text) = text_response {
            return Ok((ClaudeResponse::Text(text), assistant_message));
        }

        Err("No response generated.".to_string())
    }
}
