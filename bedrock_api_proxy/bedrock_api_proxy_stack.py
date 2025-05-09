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
        
        # Grant Lambda permission to call Bedrock
        bedrock_proxy_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:RetrieveAndGenerate", 
                    "bedrock:Retrieve",
                    "bedrock:GetInferenceProfile"
                ],
                resources=["*"]  # You can restrict this to specific models if needed
            )
        )
        
        # Create API Gateway with API key authentication
        api = apigateway.RestApi(
            self, "BedrockProxyApi",
            rest_api_name="Bedrock Proxy API",
            description="API Gateway proxy for Bedrock retrieveAndGenerate endpoint",
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
        
        # Create API Gateway resources and methods
        retrieve_and_generate = api.root.add_resource("retrieveAndGenerate")
        
        # Add POST method with Lambda integration
        retrieve_and_generate.add_method(
            "POST",
            apigateway.LambdaIntegration(
                bedrock_proxy_lambda,
                proxy=True
            ),
            api_key_required=True
        )
        
        # Output the API endpoint URL and API key
        CfnOutput(self, "ApiEndpoint", value=api.url)
        CfnOutput(self, "ApiKeyId", value=api_key.key_id)
        CfnOutput(self, "Region", value=region)
