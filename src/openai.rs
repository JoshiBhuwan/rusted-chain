use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
}

#[derive(Serialize)]
struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: serde_json::Value,
}

#[derive(Serialize, Clone)]
pub(crate) struct Message {
    pub(crate) role: String,
    pub(crate) content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(rename = "tool_call_id", skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<ToolCallResponse>>,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: MessageResponse,
}

#[derive(Deserialize)]
struct MessageResponse {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallResponse>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct ToolCallResponse {
    id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    tool_type: String,
    function: FunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
struct FunctionCall {
    name: String,
    arguments: String,
}

pub enum OpenAIResponse {
    Text(String),
    ToolCall(ToolCall),
}

pub struct ToolCall {
    pub name: String,
    pub args: serde_json::Value,
    #[allow(dead_code)] // Reserved for multi-turn tool conversations
    pub id: String,
}

pub struct OpenAI {
    api_key: String,
    model: String,
    client: Client,
    tools: Option<Vec<serde_json::Value>>,
}

impl Default for OpenAI {
    fn default() -> Self {
        dotenv::dotenv().ok();
        Self {
            api_key: env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: "gpt-4o-mini".to_string(),
            client: Client::new(),
            tools: None,
        }
    }
}

impl OpenAI {
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
            OpenAIResponse::Text(text) => Ok(text),
            OpenAIResponse::ToolCall(tool_call) => {
                // For simple invoke, we just return a message about the tool call
                Ok(format!("Request to call tool: {}", tool_call.name))
            }
        }
    }

    pub async fn invoke_with_response(&self, prompt: &str) -> Result<OpenAIResponse, String> {
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }];

        let (response, _) = self.chat(messages).await?;
        Ok(response)
    }

    pub(crate) async fn chat(
        &self,
        messages: Vec<Message>,
    ) -> Result<(OpenAIResponse, Message), String> {
        let url = "https://api.openai.com/v1/chat/completions";

        let tools = self.tools.as_ref().map(|t| {
            t.iter()
                .map(|tool| Tool {
                    tool_type: "function".to_string(),
                    function: tool.clone(),
                })
                .collect()
        });

        let request_body = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            tools,
        };

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Failed to send request: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("API Error {}: {}", status, text));
        }

        let response_body: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        if let Some(choice) = response_body.choices.first() {
            let assistant_message = Message {
                role: choice.message.role.clone(),
                content: choice.message.content.clone().unwrap_or_default(),
                name: None,
                tool_call_id: None,
                tool_calls: choice.message.tool_calls.clone(),
            };

            if let Some(tool_calls) = &choice.message.tool_calls {
                if let Some(tool_call) = tool_calls.first() {
                    let args_value: Value =
                        serde_json::from_str(&tool_call.function.arguments)
                            .unwrap_or(Value::Null);

                    return Ok((
                        OpenAIResponse::ToolCall(ToolCall {
                            name: tool_call.function.name.clone(),
                            args: args_value,
                            id: tool_call.id.clone(),
                        }),
                        assistant_message,
                    ));
                }
            }

            if let Some(content) = &choice.message.content {
                return Ok((OpenAIResponse::Text(content.clone()), assistant_message));
            }
        }

        Err("No response generated.".to_string())
    }
}
