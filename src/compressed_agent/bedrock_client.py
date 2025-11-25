"""AWS Bedrock client with token counting."""
import json
import boto3
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage tracking."""
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    total_tokens: int = Field(default=0, description="Total tokens used")


class BedrockResponse(BaseModel):
    """Bedrock API response with token usage."""
    content: str = Field(..., description="Generated content")
    usage: TokenUsage = Field(..., description="Token usage")
    model_id: str = Field(..., description="Model used")


class BedrockClient:
    """AWS Bedrock client with token counting."""

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region_name: str = "us-east-1"
    ):
        self.model_id = model_id
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        self.total_usage = TokenUsage()

    def invoke_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> BedrockResponse:
        """Invoke Bedrock model and track token usage."""
        # Prepare messages for Claude 3 format
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        # Prepare body for Claude 3
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            body["system"] = [{"type": "text", "text": system_prompt}]

        # Invoke model
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        
        # Extract content
        content = ""
        if response_body.get('content'):
            for content_block in response_body['content']:
                if content_block.get('type') == 'text':
                    content += content_block.get('text', '')

        # Extract token usage
        usage_data = response_body.get('usage', {})
        usage = TokenUsage(
            input_tokens=usage_data.get('input_tokens'),
            output_tokens=usage_data.get('output_tokens'),
            total_tokens=usage_data.get('input_tokens') + usage_data.get('output_tokens')
        )

        # Update total usage
        self.total_usage.input_tokens += usage.input_tokens
        self.total_usage.output_tokens += usage.output_tokens
        self.total_usage.total_tokens += usage.total_tokens

        return BedrockResponse(
            content=content,
            usage=usage,
            model_id=self.model_id
        )

    def get_total_usage(self) -> TokenUsage:
        """Get total token usage across all calls."""
        return self.total_usage

    def reset_usage(self):
        """Reset token usage tracking."""
        self.total_usage = TokenUsage()

