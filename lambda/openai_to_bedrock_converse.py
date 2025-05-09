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


def map_openai_to_bedrock_converse(openai_request, kb_id=None):
    """
    Maps an OpenAI API request to a Bedrock Converse request
    """
    # Extract parameters from OpenAI request
    messages = openai_request.get('messages', [])
    bedrock_model_id = openai_request.get('model', '')
    temperature = openai_request.get('temperature', 0.7)
    max_tokens = openai_request.get('max_tokens', 4096)
    top_p = openai_request.get('top_p', 0.9)
    stop_sequences = openai_request.get('stop', [])
    
    # Extract system message if present
    system_content = None
    filtered_messages = []
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            system_content = content
        elif role == 'user' or role == 'assistant':
            filtered_messages.append({
                "role": role,
                "content": [{"text": content}]
            })
    
    # Create inference config with camelCase parameter names
    inference_config = {
        "temperature": temperature,
        "topP": top_p,
        "maxTokens": max_tokens
    }
    
    # Add stop sequences if provided
    if stop_sequences:
        inference_config["stopSequences"] = stop_sequences
    
    # Create response with parameters for direct API call
    response = {
        "modelId": bedrock_model_id,
        "messages": filtered_messages,
        "inferenceConfig": inference_config
    }
    
    # Add system message if present
    if system_content:
        response["system"] = [{"text": system_content}]
    
    # Add knowledge base configuration if KB ID is provided
    if kb_id:
        response["knowledgeBaseConfig"] = {
            "knowledgeBaseId": kb_id,
            "retrievalConfig": {
                "vectorSearchConfig": {
                    "numberOfResults": 10
                }
            }
        }
    
    return response

def map_bedrock_converse_to_openai(bedrock_response):
    """
    Maps a Bedrock Converse response to an OpenAI API response
    """
    try:
        # Extract the generated text from Bedrock response
        generated_text = ""
        
        if "output" in bedrock_response and "message" in bedrock_response["output"]:
            message = bedrock_response["output"]["message"]
            if "content" in message and isinstance(message["content"], list):
                # Concatenate all text content
                for content_item in message["content"]:
                    if "text" in content_item:
                        generated_text += content_item["text"]
        
        # Create OpenAI-compatible response
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": bedrock_response.get("modelId", "bedrock-converse"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": bedrock_response.get("stopReason", "stop")
                }
            ],
            "usage": {
                "prompt_tokens": bedrock_response.get("usage", {}).get("inputTokenCount", 0),
                "completion_tokens": bedrock_response.get("usage", {}).get("outputTokenCount", 0),
                "total_tokens": (
                    bedrock_response.get("usage", {}).get("inputTokenCount", 0) + 
                    bedrock_response.get("usage", {}).get("outputTokenCount", 0)
                )
            }
        }
        
        # Add citations if available
        if "citations" in bedrock_response:
            openai_response["choices"][0]["message"]["metadata"] = {
                "citations": bedrock_response["citations"]
            }
        
        return openai_response
    
    except Exception as e:
        logger.error(f"Error mapping Bedrock Converse response to OpenAI format: {str(e)}")
        raise

def handler(event, context):
    """
    Lambda handler that converts OpenAI API requests to Bedrock Converse requests
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Get the region from the Lambda environment - automatically set by AWS
        region = os.environ.get('AWS_REGION')
        
        # Get the knowledge base ID from environment variables
        kb_id = os.environ.get('KNOWLEDGE_BASE_ID', '')
        
        # Initialize Bedrock client
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
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
        
        # Log the request body for debugging
        logger.info(f"Request body: {json.dumps(body)}")
        
        # Determine if this is a KB query or a chat completion request
        # Use the full path instead of pathParameters for more reliable endpoint detection
        endpoint_path = event.get('path', '')
        logger.info(f"Endpoint path: {endpoint_path}")
        
        # Check if the path contains the specific endpoint patterns
        use_kb = '/v2/kb/completions' in endpoint_path
        
        # Map the OpenAI request to a Bedrock Converse request
        if use_kb and kb_id:
            logger.info("Processing as KB query with Converse API")
            converse_params = map_openai_to_bedrock_converse(body, kb_id)
        else:
            logger.info("Processing as chat completion with Converse API")
            converse_params = map_openai_to_bedrock_converse(body)
        
        # Log the Bedrock request for debugging
        logger.info(f"Bedrock request: {json.dumps(converse_params)}")
        
        # Extract parameters for direct API call
        model_id = converse_params.pop("modelId")
        
        try:
            # Create a simplified request with only essential parameters
            simplified_request = {
                "modelId": model_id,
                "messages": converse_params.get("messages", [])
            }
            
            # Add system parameter if present
            if "system" in converse_params:
                simplified_request["system"] = converse_params["system"]
            
            # Add inference config if present
            if "inferenceConfig" in converse_params:
                simplified_request["inferenceConfig"] = converse_params["inferenceConfig"]
            
            # Add knowledge base config if present
            if "knowledgeBaseConfig" in converse_params:
                simplified_request["knowledgeBaseConfig"] = converse_params["knowledgeBaseConfig"]
            
            # Log the simplified request
            logger.info(f"Simplified Bedrock request: {json.dumps(simplified_request)}")
            
            # Call Bedrock Converse API with simplified parameters
            bedrock_response = bedrock_runtime.converse(**simplified_request)
            
            # Log the raw Bedrock response for debugging
            logger.info(f"Raw Bedrock response keys: {list(bedrock_response.keys())}")
            logger.info(f"Raw Bedrock response: {bedrock_response}")
            
            # For the new Converse API, the response is directly in the output field
            if "output" in bedrock_response:
                # Convert Bedrock response to OpenAI format
                openai_response = map_bedrock_converse_to_openai(bedrock_response)
                
                # Return the OpenAI-compatible response
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps(openai_response)
                }
            else:
                logger.error(f"Response does not contain 'output' key. Available keys: {list(bedrock_response.keys())}")
                return {
                    'statusCode': 500,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'error': 'Internal server error',
                        'message': f"Unexpected response format from Bedrock Converse API. Keys: {list(bedrock_response.keys())}"
                    })
                }
                response_body = json.loads(bedrock_response["response"])
                
                # Log the parsed response for debugging
                logger.info(f"Parsed Bedrock response: {json.dumps(response_body)}")
                
                # Convert Bedrock response to OpenAI format
                openai_response = map_bedrock_converse_to_openai(response_body)
                
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
