# Bedrock API Proxy with OpenAI Compatibility

This project provides an API Gateway that allows applications built for OpenAI's API to use Amazon Bedrock instead. It supports both direct Bedrock API calls and OpenAI-compatible endpoints.

## Key Features

- **OpenAI API Compatibility**: Use OpenAI-compatible endpoints to access Bedrock models
- **Knowledge Base Integration**: Query Bedrock Knowledge Bases using OpenAI-style requests
- **API Key Authentication**: Secure access with API keys
- **Model Mapping**: Automatically maps OpenAI model names to Bedrock models

## Implementation Details

The project includes:

1. **Native Bedrock Endpoint**:
   - `/retrieveAndGenerate`: Direct access to Bedrock's RetrieveAndGenerate API

2. **OpenAI-Compatible Endpoints**:
   - `/v1/chat/completions`: Maps to Bedrock's InvokeModel API
   - `/v1/kb/completions`: Maps to Bedrock's RetrieveAndGenerate API

3. **Lambda Functions**:
   - `bedrock_proxy.py`: Handles native Bedrock API calls
   - `openai_to_bedrock.py`: Translates OpenAI API requests to Bedrock format

## Model Mapping

The adapter maps OpenAI model names to Bedrock models:

- `gpt-3.5-turbo` → `anthropic.claude-3-sonnet-20240229-v1:0`
- `gpt-4` → `anthropic.claude-3-opus-20240229-v1:0`
- `gpt-4-turbo` → `anthropic.claude-3-opus-20240229-v1:0`

You can customize this mapping in the `openai_to_bedrock.py` file.

## Configuration

The OpenAI adapter Lambda function can be configured with these environment variables:

- `KNOWLEDGE_BASE_ID`: ID of your Bedrock Knowledge Base (required for KB queries)
- `MODEL_ARN`: Optional ARN of the model to use for inference

## Usage Examples

### Chat Completions (OpenAI-compatible)

```bash
curl -X POST \
  https://your-api-id.execute-api.REGION.amazonaws.com/prod/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: YOUR_API_KEY_VALUE' \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about AWS Bedrock."}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### Knowledge Base Queries (OpenAI-compatible)

```bash
curl -X POST \
  https://your-api-id.execute-api.REGION.amazonaws.com/prod/v1/kb/completions \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: YOUR_API_KEY_VALUE' \
  -d '{
    "messages": [
      {"role": "user", "content": "What information do you have about AWS Bedrock?"}
    ],
    "temperature": 0.3,
    "max_tokens": 1000
  }'
```

## Benefits

- **Simplified Integration**: Use existing OpenAI-compatible libraries and tools with Bedrock
- **Reduced Vendor Lock-in**: Switch between OpenAI and Bedrock without changing application code
- **Cost Optimization**: Take advantage of Bedrock's pricing model
- **Enhanced Privacy**: Keep your data within AWS infrastructure
