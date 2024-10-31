from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential,AzureCliCredential
import os

def get_conn():
    keyVaultName = os.environ["KEY_VAULT_NAME"]
    KVUri = f"https://{keyVaultName}.vault.azure.net/"

    credential = DefaultAzureCredential(additionally_allowed_tenants=['*'])
    client = SecretClient(vault_url=KVUri, credential=credential)

    return client