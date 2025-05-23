# Bedrock API Proxy with OpenAI Compatibility

This project creates an API Gateway with API key authentication that proxies requests to Amazon Bedrock endpoints. It supports both native Bedrock API calls and OpenAI-compatible API calls, allowing third-party applications built for OpenAI to use Amazon Bedrock instead.

## Architecture

- **API Gateway**: Provides REST API endpoints with API key authentication
- **Lambda Functions**: 
  - Native Bedrock proxy for retrieveAndGenerate endpoint
  - OpenAI-compatible adapter that translates OpenAI API calls to Bedrock API calls
- **IAM Permissions**: Grants the Lambda functions permission to call Bedrock
- **Supported Endpoints**:
  - Native Bedrock: `/retrieveAndGenerate`
  - OpenAI Chat Completions: `/v1/chat/completions` (maps to Bedrock InvokeModel)
  - OpenAI KB Completions: `/v1/kb/completions` (maps to Bedrock RetrieveAndGenerate)
  - OpenAI v2 API: `/v2/chat/completions` (maps to Bedrock InvokeModel with enhanced features)

## Prerequisites

- AWS CLI configured with appropriate credentials
- Python 3.11 or higher
- uv (Python package manager)
- AWS CDK v2

## Setup

1. Activate the virtual environment:
   ```
   cd bedrock-api-proxy
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```
   uv pip install -r requirements.txt
   uv pip install -r requirements-dev.txt
   ```

3. (Optional) Configure the deployment region:
   
   Edit the `cdk.json` file and update the `region` value in the `context` section:
   ```json
   "context": {
     "region": "ap-northeast-1",
     ...
   }
   ```

4. Deploy the stack:
   ```
   cdk deploy
   ```
   
   The stack will be deployed using the account from your current AWS profile and the region specified in `cdk.json`.

5. After deployment, note the API endpoint URL and API key ID from the outputs.

6. Retrieve the API key value:
   ```
   aws apigateway get-api-key --api-key YOUR_API_KEY_ID --include-value
   ```

## Usage

### Native Bedrock API

To call the native Bedrock retrieveAndGenerate endpoint:

```bash
curl -X POST \
  https://your-api-id.execute-api.REGION.amazonaws.com/prod/retrieveAndGenerate \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: YOUR_API_KEY_VALUE' \
  -d '{
    "input": {
      "text": "{replace-with-your-query}"
    },
    "retrieveAndGenerateConfiguration": {
		"knowledgeBaseConfiguration": {
			"generationConfiguration": {
					"inferenceConfig": {
					"textInferenceConfig": {
						"maxTokens": 4000,
						"stopSequences": [
							"Human:"
						],
						"temperature": 0.3,
						"topP": 0.1
					}
				},
				"promptTemplate": {
					"textPromptTemplate": "You are a question answering agent. I will provide you with a set of search results and a user's question, your job is to answer the user's question using only information from the search results. If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion. \n Here are the search results in numbered order:\n<context>\n$search_results$\n</context>\nHere is the user's question:\n<question>\n$query$\n</question>\n$output_format_instructions$\nAssistant:"
				}
			},
			"knowledgeBaseId": "{replace-with-your-kb-id}",
			"modelArn": "{replace-with-your-Inference profile ID-in-Cross-region inference-Inference profiles}",
			"retrievalConfiguration": {
				"vectorSearchConfiguration": {
					"numberOfResults": 10
				}
			}
		},
      "type": "KNOWLEDGE_BASE"
    }
  }'
```

### OpenAI-Compatible Chat Completions API

To use the OpenAI-compatible chat completions endpoint:

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

### OpenAI-Compatible Knowledge Base API

To use the OpenAI-compatible knowledge base endpoint:

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

**Note:** For the KB endpoint, you need to set the `KNOWLEDGE_BASE_ID` and optionally `MODEL_ARN` environment variables in the Lambda function.

### OpenAI-Compatible v2 Chat Completions API

The v2 endpoint provides enhanced features and compatibility with newer OpenAI API specifications:

```bash
curl -X POST \
  https://your-api-id.execute-api.REGION.amazonaws.com/prod/v2/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: YOUR_API_KEY_VALUE' \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about AWS Bedrock."}
    ],
    "temperature": 0.7,
    "max_tokens": 500,
    "stream": true
  }'
```

The v2 endpoint supports:
- Streaming responses with `stream: true`
- Enhanced model mapping
- Additional OpenAI-compatible parameters
- Improved error handling and response formatting

## Environment Variables

The OpenAI adapter Lambda function uses these environment variables:

- `KNOWLEDGE_BASE_ID`: The ID of your Bedrock Knowledge Base (required for KB queries)
- `MODEL_ARN`: Optional ARN of the model to use for inference

You can set these during deployment or update them later in the Lambda console.

## Cleanup

To remove all resources:

```
cdk destroy
```
