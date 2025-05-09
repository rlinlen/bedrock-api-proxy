#!/usr/bin/env python3
import os
import aws_cdk as cdk
from bedrock_api_proxy.bedrock_api_proxy_stack import BedrockApiProxyStack

app = cdk.App()

# Get the region from context (cdk.json) or use ap-northeast-1 as fallback
account = app.node.try_get_context("accountId") or None
region = app.node.try_get_context("region") or "ap-northeast-1"

BedrockApiProxyStack(app, "BedrockApiProxyStack",
    # This will use the account from the current AWS profile
    # and the region from cdk.json context
    env=cdk.Environment(
        account=account,
        region=region
    ),
)

app.synth()
