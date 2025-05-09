import json
import os
import boto3
import logging
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    """
    Lambda handler that proxies requests to Bedrock retrieveAndGenerate endpoint
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Get the region from the Lambda environment
        region = os.environ.get('AWS_REGION', 'ap-northeast-1')
        
        # Initialize Bedrock client with the current region
        bedrock_runtime = boto3.client(
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
        
        # Call Bedrock retrieveAndGenerate API
        response = bedrock_runtime.retrieve_and_generate(
            input=body.get('input', {}),
            retrieveAndGenerateConfiguration=body.get('retrieveAndGenerateConfiguration', {})
        )
        
        # Process and return the response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'output': response.get('output', {}),
                'citations': response.get('citations', []),
                'usage': response.get('usage', {})
            })
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
