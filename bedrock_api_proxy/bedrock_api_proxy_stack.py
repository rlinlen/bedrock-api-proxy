from aws_cdk import (
    Duration,
    Stack,
    aws_apigateway as apigateway,
    aws_lambda as lambda_,
    aws_iam as iam,
    CfnOutput,
)
from constructs import Construct

class BedrockApiProxyStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Get the region from the environment
        region = self.region

        # Create Lambda function for Bedrock proxy
        bedrock_proxy_lambda = lambda_.Function(
            self, "BedrockProxyLambda",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="bedrock_proxy.handler",
            code=lambda_.Code.from_asset("lambda"),
            timeout=Duration.seconds(60),
            memory_size=256,
            environment={
                "BEDROCK_ENDPOINT": f"https://bedrock-agent-runtime.{region}.amazonaws.com",
            }
        )
        
        # Create Lambda function for OpenAI to Bedrock adapter
        openai_adapter_lambda = lambda_.Function(
            self, "OpenAIAdapterLambda",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="openai_to_bedrock.handler",
            code=lambda_.Code.from_asset("lambda"),
            timeout=Duration.seconds(60),
            memory_size=256,
            environment={
                # These can be overridden during deployment
                "KNOWLEDGE_BASE_ID": "",  # Optional: Default KB ID
            }
        )
        
        # Create Lambda function for OpenAI to Bedrock adapter using Converse API
        openai_converse_adapter_lambda = lambda_.Function(
            self, "OpenAIConverseAdapterLambda",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="openai_to_bedrock_converse.handler",
            code=lambda_.Code.from_asset("lambda"),
            timeout=Duration.seconds(60),
            memory_size=256,
            environment={
                # These can be overridden during deployment
                "KNOWLEDGE_BASE_ID": "",  # Optional: Default KB ID
            }
        )
        
        # Grant Lambda permissions to call Bedrock
        for lambda_function in [bedrock_proxy_lambda, openai_adapter_lambda, openai_converse_adapter_lambda]:
            lambda_function.add_to_role_policy(
                iam.PolicyStatement(
                    actions=[
                        "bedrock:InvokeModel",
                        "bedrock:RetrieveAndGenerate", 
                        "bedrock:Retrieve",
                        "bedrock:GetInferenceProfile",
                        "bedrock:Converse"
                    ],
                    resources=["*"]  # You can restrict this to specific models if needed
                )
            )
        
        # Create API Gateway with API key authentication
        api = apigateway.RestApi(
            self, "BedrockProxyApi",
            rest_api_name="Bedrock Proxy API",
            description="API Gateway proxy for Bedrock endpoints with OpenAI compatibility",
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=apigateway.Cors.ALL_ORIGINS,
                allow_methods=apigateway.Cors.ALL_METHODS
            )
        )
        
        # Create API key and usage plan
        api_key = api.add_api_key("BedrockProxyApiKey")
        
        plan = api.add_usage_plan("BedrockProxyUsagePlan",
            name="Standard",
            throttle=apigateway.ThrottleSettings(
                rate_limit=20,
                burst_limit=30
            )
        )
        
        plan.add_api_key(api_key)
        plan.add_api_stage(
            stage=api.deployment_stage
        )
        
        # Create API Gateway resources and methods for native Bedrock endpoint
        retrieve_and_generate = api.root.add_resource("retrieveAndGenerate")
        
        # Add POST method with Lambda integration for native Bedrock endpoint
        retrieve_and_generate.add_method(
            "POST",
            apigateway.LambdaIntegration(
                bedrock_proxy_lambda,
                proxy=True
            ),
            api_key_required=True
        )
        
        # Create OpenAI-compatible endpoints
        openai = api.root.add_resource("v1")
        
        # Chat completions endpoint (maps to Bedrock InvokeModel)
        chat_completions = openai.add_resource("chat").add_resource("completions")
        chat_completions.add_method(
            "POST",
            apigateway.LambdaIntegration(
                openai_adapter_lambda,
                proxy=True
            ),
            api_key_required=True
        )
        
        # Knowledge base endpoint (maps to Bedrock RetrieveAndGenerate)
        kb_completions = openai.add_resource("kb").add_resource("completions")
        kb_completions.add_method(
            "POST",
            apigateway.LambdaIntegration(
                openai_adapter_lambda,
                proxy=True,
                request_parameters={
                    "integration.request.path.proxy": "'kb'"
                }
            ),
            api_key_required=True,
            request_parameters={
                "method.request.path.proxy": True
            }
        )
        
        # Create OpenAI-compatible endpoints with Converse API
        openai_converse = api.root.add_resource("v2")
        
        # Chat completions endpoint using Converse API
        chat_completions_converse = openai_converse.add_resource("chat").add_resource("completions")
        chat_completions_converse.add_method(
            "POST",
            apigateway.LambdaIntegration(
                openai_converse_adapter_lambda,
                proxy=True
            ),
            api_key_required=True
        )
        
        # Knowledge base endpoint using Converse API
        kb_completions_converse = openai_converse.add_resource("kb").add_resource("completions")
        kb_completions_converse.add_method(
            "POST",
            apigateway.LambdaIntegration(
                openai_converse_adapter_lambda,
                proxy=True
            ),
            api_key_required=True
        )
        
        # Output the API endpoint URL and API key
        CfnOutput(self, "ApiEndpoint", value=api.url)
        CfnOutput(self, "ApiKeyId", value=api_key.key_id)
        CfnOutput(self, "Region", value=region)
        CfnOutput(self, "OpenAIChatEndpoint", value=f"{api.url}v1/chat/completions")
        CfnOutput(self, "OpenAIKbEndpoint", value=f"{api.url}v1/kb/completions")
        CfnOutput(self, "OpenAIConverseEndpoint", value=f"{api.url}v2/chat/completions")
        CfnOutput(self, "OpenAIKbConverseEndpoint", value=f"{api.url}v2/kb/completions")
