# Bedrock API Proxy

This project creates an API Gateway with API key authentication that proxies requests to the Amazon Bedrock retrieveAndGenerate endpoint. This will be able to faciliate third party API call that wants to utilize the bedrock knowledge base function, e.g. n8n calling bedrock KB using HTTP Request node.

## Architecture

- **API Gateway**: Provides a REST API endpoint with API key authentication
- **Lambda Function**: Proxies requests to the Bedrock retrieveAndGenerate endpoint
- **IAM Permissions**: Grants the Lambda function permission to call Bedrock

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

To call the API, use the following format:

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
			"modelArn": "{replace-with-your-Inference profile ID-in-
Cross-region inference-Inference profiles}",
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

Refer to https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html#kb-test-config-sysprompt for more format info.

## Cleanup

To remove all resources:

```
cdk destroy
```
