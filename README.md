# SAP AI Core Extension for OpenWebUI

This extension integrates SAP AI Core with OpenWebUI, allowing users to leverage SAP AI Core's deployments within the OpenWebUI interface. It utilizes the pipeline extensibility feature of OpenWebUI to add SAP AI Core as an additional option for AI model inference.

For more details on how to use Pipelines in OpenWebUI, please refer to the [OpenWebUI Pipelines documentation](https://docs.openwebui.com/pipelines/#integration-examples).

## Overview

This extension enables OpenWebUI users to:

1. Connect to SAP AI Core
2. List available deployments from SAP AI Core Instance
3. Use SAP AI Core deployments as models within OpenWebUI
4. Access a wider range of AI models supported by SAP AI Core

## Configuration

To use this extension, you need to configure the following parameters in the Pipelines section of the OpenWebUI:

```
Ai Core Client Id: Your SAP AI Core client ID
Ai Core Client Secret: Your SAP AI Core client secret
Ai Core Auth Url: The token URL for authentication (e.g., "https://<account>.authentication.sap.hana.ondemand.com")
Ai Core Base Url: The base URL for SAP AI Core API (e.g., "https://api.ai.<region>.aws.ml.hana.ondemand.com/v2")
```

These parameters are essential for authenticating and connecting to your SAP AI Core instance.

## Usage

1. Ensure that you have created deployments inside your SAP AI Core instance.
2. Import the SAP AI Core pipeline into OpenWebUI:
   - Go to the GitHub project and open the `sapaicore.py` file.
   - Click on the "Raw" button to view the raw file content.
   - Copy the URL of the raw file.
   - In OpenWebUI, navigate to Admin Panel -> Settings -> Pipelines.
   - Paste the copied URL into the "Install from Github URL" field.
   - Click the download button to import the pipeline.
3. Once OpenWebUI finishes importing, it will prompt you to configure the connection:
   - Fill in the required fields with your SAP AI Core credentials and endpoints.
4. After configuration, all available and running deployments in SAP AI Core that are supported by the generative-ai-hub-sdk will be automatically listed as models within OpenWebUI.
5. Select the desired SAP AI Core deployment from the model list in OpenWebUI to use it for inference.
6. To check which models are currently supported, visit the [SAP Generative AI Hub SDK documentation](https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_reference/README_sphynx.html#llm-models).

## Important Notes

- This extension requires an active SAP AI Core account and properly configured deployments.
- Only deployments with a "RUNNING" status in SAP AI Core will be available in OpenWebUI.
- The extension now uses the generative-ai-hub-sdk and langchain, increasing support for various AI models.
- Supported models include, but are not limited to, GPT and Anthropic API models. For a complete list of supported models, refer to the SAP Generative AI Hub SDK documentation.

## Future Enhancements

Stay tuned for updates!