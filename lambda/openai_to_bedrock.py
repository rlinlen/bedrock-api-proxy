import json
import os
import boto3
import logging
import time
import uuid
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def format_messages_for_claude(messages):
    """
    Format OpenAI-style messages into Claude prompt format
    """
    formatted_prompt = ""
    
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        if role == 'system':
            formatted_prompt += f"{content}\n\n"
        elif role == 'user':
            formatted_prompt += f"Human: {content}\n\n"
        elif role == 'assistant':
            formatted_prompt += f"Assistant: {content}\n\n"
        # Ignore other roles like 'function' for simplicity
    
    # Add the final assistant prefix to prompt Claude to respond
    formatted_prompt += "Assistant: "
    
    return formatted_prompt

def map_openai_to_bedrock_chat(openai_request):
    """
    Maps an OpenAI chat completion API request to a Bedrock InvokeModel request
    """
    # Extract parameters from OpenAI request
    messages = openai_request.get('messages', [])
    bedrock_model_id = openai_request.get('model', "")
    temperature = openai_request.get('temperature', 0.7)
    max_tokens = openai_request.get('max_tokens', 4096)
    top_p = openai_request.get('top_p', 0.9)
    stop_sequences = openai_request.get('stop', [])

    # For Claude 3 models, use the messages API format
    # Convert OpenAI messages to Claude messages format
    claude_messages = []
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            claude_messages.append({"role": "user", "content": content})
            claude_messages.append({"role": "assistant", "content": "I'll follow these instructions carefully."})
        elif role == 'user':
            claude_messages.append({"role": "user", "content": content})
        elif role == 'assistant':
            claude_messages.append({"role": "assistant", "content": content})
    
    bedrock_request = {
        "modelId": bedrock_model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": claude_messages
        }
    }
    
    # Add stop sequences if provided
    if stop_sequences:
        bedrock_request["body"]["stop_sequences"] = stop_sequences
    
    return bedrock_request

def map_bedrock_chat_to_openai(bedrock_response, model_name):
    """
    Maps a Bedrock InvokeModel response to an OpenAI API response
    """
    try:
        response_body = json.loads(bedrock_response.get('body').read().decode('utf-8'))
        
        # Extract the generated text based on model type
        generated_text = ""
        
        if "anthropic.claude" in model_name:
            # Claude 3 response format (messages API)
            if 'content' in response_body and isinstance(response_body['content'], list) and len(response_body['content']) > 0:
                generated_text = response_body['content'][0].get('text', '')
            else:
                # Fallback for older Claude format
                generated_text = response_body.get('completion', '')
        else:
            # Generic format for other models
            generated_text = response_body.get('completion', '')
        
        # Create OpenAI-compatible response
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": response_body.get('usage', {}).get('input_tokens', 0),
                "completion_tokens": response_body.get('usage', {}).get('output_tokens', 0),
                "total_tokens": (
                    response_body.get('usage', {}).get('input_tokens', 0) + 
                    response_body.get('usage', {}).get('output_tokens', 0)
                )
            }
        }
        
        return openai_response
    
    except Exception as e:
        logger.error(f"Error mapping Bedrock chat response to OpenAI format: {str(e)}")
        raise

def map_openai_to_bedrock_kb(openai_request, kb_id):
    """
    Maps an OpenAI API request to a Bedrock retrieveAndGenerate request
    """
    # Extract parameters from OpenAI request
    messages = openai_request.get('messages', [])
    temperature = openai_request.get('temperature', 0.7)
    max_tokens = openai_request.get('max_tokens', 4000)
    top_p = openai_request.get('top_p', 0.9)
    stop_sequences = openai_request.get('stop', [])
    
    # Get model ARN from the request if provided
    model_arn = openai_request.get('model_arn', None)
    
    # Combine all user messages into a single query
    query = ""
    for message in messages:
        if message.get('role') == 'user':
            query += message.get('content', '') + " "
    
    query = query.strip()
    
    # Create Bedrock retrieveAndGenerate request
    bedrock_request = {
        "input": {
            "text": query
        },
        "retrieveAndGenerateConfiguration": {
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": 10
                    }
                },
                "generationConfiguration": {
                    "inferenceConfig": {
                        "textInferenceConfig": {
                            "maxTokens": max_tokens,
                            "temperature": temperature,
                            "topP": top_p
                        }
                    },
                    "promptTemplate": {
                        "textPromptTemplate": "You are a question answering agent. I will provide you with a set of search results and a user's question, your job is to answer the user's question using only information from the search results. If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion. \n Here are the search results in numbered order:\n<context>\n$search_results$\n</context>\nHere is the user's question:\n<question>\n$query$\n</question>\n$output_format_instructions$\nAssistant:"
                    }
                }
            },
            "type": "KNOWLEDGE_BASE"
        }
    }
    
    # Add model ARN if provided
    if model_arn:
        bedrock_request["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["modelArn"] = model_arn
    
    # Add stop sequences if provided
    if stop_sequences:
        bedrock_request["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["generationConfiguration"]["inferenceConfig"]["textInferenceConfig"]["stopSequences"] = stop_sequences
    
    return bedrock_request

def map_bedrock_kb_to_openai(bedrock_response):
    """
    Maps a Bedrock retrieveAndGenerate response to an OpenAI API response
    """
    try:
        # Extract the generated text from Bedrock response
        generated_text = bedrock_response.get('output', {}).get('text', '')
        
        # Extract citations if available
        citations = bedrock_response.get('citations', [])
        
        # Create OpenAI-compatible response
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "bedrock-kb-proxy",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": bedrock_response.get('usage', {}).get('inputTokenCount', 0),
                "completion_tokens": bedrock_response.get('usage', {}).get('outputTokenCount', 0),
                "total_tokens": (
                    bedrock_response.get('usage', {}).get('inputTokenCount', 0) + 
                    bedrock_response.get('usage', {}).get('outputTokenCount', 0)
                )
            }
        }
        
        # Add citations as metadata if available
        if citations:
            openai_response["choices"][0]["message"]["metadata"] = {
                "citations": citations
            }
        
        return openai_response
    
    except Exception as e:
        logger.error(f"Error mapping Bedrock KB response to OpenAI format: {str(e)}")
        raise

def handler(event, context):
    """
    Lambda handler that converts OpenAI API requests to Bedrock requests
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Get the region from the Lambda environment - automatically set by AWS
        region = os.environ.get('AWS_REGION')
        
        # Get the knowledge base ID from environment variables
        kb_id = os.environ.get('KNOWLEDGE_BASE_ID', '')
        
        # Initialize Bedrock clients
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=region
        )
        
        bedrock_agent_runtime = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=region
        )
        
        # Extract the request body
        if 'body' in event:
            try:
                body = json.loads(event['body'])
            except json.JSONDecodeError:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({'error': 'Invalid JSON in request body'})
                }
        else:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Missing request body'})
            }
        
        # Determine if this is a KB query or a chat completion request
        # Use the full path instead of pathParameters for more reliable endpoint detection
        endpoint_path = event.get('path', '')
        logger.info(f"Endpoint path: {endpoint_path}")
        
        # Check if the path contains the specific endpoint patterns
        if '/v1/kb/completions' in endpoint_path and kb_id:
            # Handle as a KB query
            logger.info("Processing as KB query")
            bedrock_request = map_openai_to_bedrock_kb(body, kb_id)
            
            # Call Bedrock retrieveAndGenerate API
            bedrock_response = bedrock_agent_runtime.retrieve_and_generate(
                input=bedrock_request.get('input', {}),
                retrieveAndGenerateConfiguration=bedrock_request.get('retrieveAndGenerateConfiguration', {})
            )
            
            # Convert Bedrock response to OpenAI format
            openai_response = map_bedrock_kb_to_openai(bedrock_response)
            
        elif '/v1/chat/completions' in endpoint_path:
            # Handle as a chat completion request
            logger.info("Processing as chat completion")
            bedrock_request = map_openai_to_bedrock_chat(body)
            
            # Extract model ID and body
            model_id = bedrock_request.pop('modelId')
            request_body = json.dumps(bedrock_request.pop('body'))
            
            # Call Bedrock InvokeModel API
            bedrock_response = bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType=bedrock_request.get('contentType', 'application/json'),
                accept=bedrock_request.get('accept', 'application/json'),
                body=request_body
            )
            
            # Convert Bedrock response to OpenAI format
            openai_response = map_bedrock_chat_to_openai(bedrock_response, model_id)
        else:
            logger.error(f"Unsupported endpoint: {endpoint_path}")
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': f'Unsupported endpoint: {endpoint_path}'})
            }
        
        # Return the OpenAI-compatible response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(openai_response)
        }
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        
        logger.error(f"Bedrock API error: {error_code} - {error_message}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f"Bedrock API error: {error_code}",
                'message': error_message
            })
        }
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }
