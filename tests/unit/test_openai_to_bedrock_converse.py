import json
import unittest
from unittest import mock
import boto3
import os
import sys
from botocore.stub import Stubber

# Add the Lambda function directory to the path so we can import it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lambda')))

# Import the Lambda handler
import openai_to_bedrock_converse

class TestBedrockConverseAdapter(unittest.TestCase):
    def setUp(self):
        # Create a mock for the boto3 client
        self.bedrock_runtime_mock = mock.MagicMock()
        # Set up environment variables
        os.environ['AWS_REGION'] = 'ap-northeast-1'
        os.environ['KNOWLEDGE_BASE_ID'] = 'test-kb-id'
        
    @mock.patch('boto3.client')
    def test_chat_completion_endpoint(self, mock_boto3_client):
        # Set up the mock to return our mock client
        mock_boto3_client.return_value = self.bedrock_runtime_mock
        
        # Mock the converse response
        mock_response = {
            "response": json.dumps({
                "output": {
                    "message": {
                        "content": [
                            {"text": "AWS Bedrock is a fully managed service that offers a choice of high-performing foundation models."}
                        ]
                    }
                },
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 20
                }
            })
        }
        self.bedrock_runtime_mock.converse.return_value = mock_response
        
        # Create a test event
        event = {
            "path": "/v2/chat/completions",
            "body": json.dumps({
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Tell me about AWS Bedrock."}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            })
        }
        
        # Call the handler
        response = openai_to_bedrock_converse.handler(event, {})
        
        # Verify the response
        self.assertEqual(response['statusCode'], 200)
        
        # Parse the response body
        response_body = json.loads(response['body'])
        
        # Check that the response has the expected structure
        self.assertIn('choices', response_body)
        self.assertEqual(len(response_body['choices']), 1)
        self.assertEqual(response_body['choices'][0]['message']['role'], 'assistant')
        self.assertIn('AWS Bedrock', response_body['choices'][0]['message']['content'])
        
        # Verify that the boto3 client was called with the right parameters
        mock_boto3_client.assert_called_with(
            service_name='bedrock-runtime',
            region_name='ap-northeast-1'
        )
        
        # Verify that converse was called with the right model ID
        call_args = self.bedrock_runtime_mock.converse.call_args[1]
        self.assertIn('modelId', call_args)
        self.assertIn('messages', call_args)
        self.assertIn('inferenceConfig', call_args)
        self.assertIn('system', call_args)
        
    @mock.patch('boto3.client')
    def test_kb_completion_endpoint(self, mock_boto3_client):
        # Set up the mock to return our mock client
        mock_boto3_client.return_value = self.bedrock_runtime_mock
        
        # Mock the converse response
        mock_response = {
            "response": json.dumps({
                "output": {
                    "message": {
                        "content": [
                            {"text": "Based on the knowledge base, AWS Bedrock is a fully managed service."}
                        ]
                    }
                },
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 15
                },
                "citations": [
                    {
                        "retrievedReferences": [
                            {
                                "content": {"text": "AWS Bedrock documentation"},
                                "location": {"s3Location": {"uri": "s3://bucket/key"}}
                            }
                        ]
                    }
                ]
            })
        }
        self.bedrock_runtime_mock.converse.return_value = mock_response
        
        # Create a test event
        event = {
            "path": "/v2/kb/completions",
            "body": json.dumps({
                "messages": [
                    {"role": "user", "content": "What information do you have about AWS Bedrock?"}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            })
        }
        
        # Call the handler
        response = openai_to_bedrock_converse.handler(event, {})
        
        # Verify the response
        self.assertEqual(response['statusCode'], 200)
        
        # Parse the response body
        response_body = json.loads(response['body'])
        
        # Check that the response has the expected structure
        self.assertIn('choices', response_body)
        self.assertEqual(len(response_body['choices']), 1)
        self.assertEqual(response_body['choices'][0]['message']['role'], 'assistant')
        self.assertIn('AWS Bedrock', response_body['choices'][0]['message']['content'])
        
        # Verify that the boto3 client was called with the right parameters
        mock_boto3_client.assert_called_with(
            service_name='bedrock-runtime',
            region_name='ap-northeast-1'
        )
        
        # Verify that converse was called with the right parameters
        call_args = self.bedrock_runtime_mock.converse.call_args[1]
        self.assertIn('modelId', call_args)
        self.assertIn('messages', call_args)
        self.assertIn('inferenceConfig', call_args)
        self.assertIn('knowledgeBaseConfig', call_args)

if __name__ == '__main__':
    unittest.main()
