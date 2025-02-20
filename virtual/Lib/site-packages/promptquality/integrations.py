from json import dumps
from typing import Dict, List, Optional, Union

from pydantic import SecretStr

from promptquality.constants.integrations import IntegrationName
from promptquality.set_config_module import set_config
from promptquality.types.config import Config
from promptquality.types.run import (
    AzureAuthenticationType,
    AzureModelDeployment,
    CreateIntegrationRequest,
)


def add_openai_integration(
    api_key: str, organization_id: Optional[str] = None, config: Optional[Config] = None
) -> None:
    """
    Add an OpenAI integration to your Galileo account.

    If you add an integration while one already exists, the new integration will
    overwrite the old one.

    Parameters
    ----------
    api_key : str
        Your OpenAI API key.
    organization_id : Optional[str], optional
        Organization ID, if you want to include it in OpenAI requests, by default None
    config : Optional[Config], optional
        Config to use, by default None which translates to the config being set
        automatically.
    """
    config = config or set_config()
    config.api_client.put_integration(
        integration_request=CreateIntegrationRequest(
            api_key=SecretStr(api_key), name=IntegrationName.openai, organization_id=organization_id
        ),
    )


def add_azure_integration(
    api_key: Union[str, Dict[str, str]],
    endpoint: str,
    authentication_type: AzureAuthenticationType = AzureAuthenticationType.api_key,
    authentication_scope: Optional[str] = None,
    available_deployments: Optional[List[AzureModelDeployment]] = None,
    headers: Optional[Dict[str, str]] = None,
    proxy: Optional[bool] = None,
    config: Optional[Config] = None,
) -> None:
    """
    Add an Azure integration to your Galileo account.

    If you add an integration while one already exists, the new integration will
    overwrite the old one.

    Parameters
    ----------
    api_key : str
        Azure authentication key. This can be one of:
        1. Your Azure API key. If you provide this, the authentication type should be
        `AzureAuthenticationType.api_key`.
        2. A dictionary containing the Azure Entra credentials with ID and secret. If
        you use this, `AZURE_CLIENT_ID`,
        `AZURE_CLIENT_SECRET` and `AZURE_TENANT_ID` are expected to be included and the authentication type should be `AzureAuthenticationType.client_secret`.
        3. A dictionary containing the Azure Entra credentials with username and password. If
        you use this, `AZURE_CLIENT_ID`, `AZURE_USERNAME` and `AZURE_PASSWORD` are expected to be included and the authentication type should be `AzureAuthenticationType.username_password`.
    endpoint : str
        The endpoint to use for the Azure API.
    authentication_type : AzureAuthenticationType, optional
        The type of authentication to use, by default AzureAuthenticationType.api_key.
    authentication_scope : Optional[str], optional
        The scope to use for authentication with Azure Entra, by default None, which translates to the default scope
        for Azure Cognitive Services (https://cognitiveservices.azure.com/.default).
    available_deployments : Optional[List[AzureModelDeployment]], optional
        The available deployments for the model. If provided, we won't try to get it from Azure directly. This list should contain values with keys `model` and `id` where the values match the model ID [1] and `id` matches the deployment ID, by default None.
        [1]: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models
    headers : Optional[Dict[str, str]], optional
        Headers to use for making requests to Azure, by default None.
    proxy : Optional[bool], optional
        Whether the endpoint provided is a proxy endpoint. If your endpoint doesn't contain `azure` in the URL, it is likely a proxy, by default None which translates to False.
    config : Optional[Config], optional
        Config to use, by default None which translates to the config being set
        automatically.
    """
    config = config or set_config()
    api_key = dumps(api_key) if isinstance(api_key, dict) else api_key
    config.api_client.put_integration(
        integration_request=CreateIntegrationRequest(
            api_key=SecretStr(api_key),
            name=IntegrationName.azure,
            endpoint=endpoint,
            authentication_type=authentication_type,
            authentication_scope=authentication_scope,
            available_deployments=available_deployments,
            headers=headers,
            proxy=proxy,
        ),
    )
