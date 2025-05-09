import aws_cdk as core
import aws_cdk.assertions as assertions

from bedrock_api_proxy.bedrock_api_proxy_stack import BedrockApiProxyStack

# example tests. To run these tests, uncomment this file along with the example
# resource in bedrock_api_proxy/bedrock_api_proxy_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = BedrockApiProxyStack(app, "bedrock-api-proxy")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
