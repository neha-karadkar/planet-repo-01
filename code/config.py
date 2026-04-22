
import os
import logging
from dotenv import load_dotenv

# Load .env file FIRST before any os.getenv() calls
load_dotenv()

class Config:
    _kv_secrets = {}

    # Key Vault secret mapping (only relevant entries for this agent)
    KEY_VAULT_SECRET_MAP = [
        # LLM API keys
        ("AZURE_OPENAI_API_KEY", "openai-secrets.gpt-4.1"),
        ("AZURE_OPENAI_API_KEY", "openai-secrets.azure-key"),
        # Azure Content Safety
        ("AZURE_CONTENT_SAFETY_ENDPOINT", "azure-content-safety-secrets.azure_content_safety_endpoint"),
        ("AZURE_CONTENT_SAFETY_KEY", "azure-content-safety-secrets.azure_content_safety_key"),
        # Observability DB
        ("OBS_AZURE_SQL_SERVER", "agentops-secrets.obs_sql_endpoint"),
        ("OBS_AZURE_SQL_DATABASE", "agentops-secrets.obs_azure_sql_database"),
        ("OBS_AZURE_SQL_PORT", "agentops-secrets.obs_port"),
        ("OBS_AZURE_SQL_USERNAME", "agentops-secrets.obs_sql_username"),
        ("OBS_AZURE_SQL_PASSWORD", "agentops-secrets.obs_sql_password"),
        ("OBS_AZURE_SQL_SCHEMA", "agentops-secrets.obs_azure_sql_schema"),
    ]

    # Models that do not support temperature/max_tokens
    _MAX_TOKENS_UNSUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1-chat", "o1", "o1-mini", "o1-preview",
        "o3", "o3-mini", "o3-pro", "o4-mini"
    }
    _TEMPERATURE_UNSUPPORTED = {
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1-chat", "o1", "o1-mini", "o1-preview",
        "o3", "o3-mini", "o3-pro", "o4-mini"
    }

    @classmethod
    def _load_keyvault_secrets(cls):
        """Load secrets from Azure Key Vault if enabled and URI is set."""
        if not getattr(cls, "USE_KEY_VAULT", False):
            return {}
        if not getattr(cls, "KEY_VAULT_URI", ""):
            return {}

        AZURE_USE_DEFAULT_CREDENTIAL = getattr(cls, "AZURE_USE_DEFAULT_CREDENTIAL", False)
        try:
            if AZURE_USE_DEFAULT_CREDENTIAL:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
            else:
                from azure.identity import ClientSecretCredential
                tenant_id = os.getenv("AZURE_TENANT_ID", "")
                client_id = os.getenv("AZURE_CLIENT_ID", "")
                client_secret = os.getenv("AZURE_CLIENT_SECRET", "")
                if not (tenant_id and client_id and client_secret):
                    logging.warning("Service Principal credentials incomplete. Key Vault access will fail.")
                    return {}
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
            from azure.keyvault.secrets import SecretClient
            client = SecretClient(vault_url=cls.KEY_VAULT_URI, credential=credential)
        except Exception as e:
            logging.warning(f"Failed to initialize Azure Key Vault client: {e}")
            return {}

        import json
        secrets_map = getattr(cls, "KEY_VAULT_SECRET_MAP", [])
        by_secret = {}
        for attr, ref in secrets_map:
            if "." in ref:
                secret_name, json_key = ref.split(".", 1)
            else:
                secret_name, json_key = ref, None
            by_secret.setdefault(secret_name, []).append((attr, json_key))

        kv_secrets = {}
        for secret_name, refs in by_secret.items():
            try:
                secret = client.get_secret(secret_name)
                if not secret or not secret.value:
                    logging.debug(f"Key Vault: secret '{secret_name}' is empty or missing")
                    continue
                raw_value = secret.value.lstrip('\ufeff')
                has_json_key = any(json_key is not None for _, json_key in refs)
                if has_json_key:
                    try:
                        data = json.loads(raw_value)
                    except Exception:
                        # Try to repair malformed JSON
                        try:
                            inner = raw_value.strip().strip('{}')
                            result = {}
                            for part in inner.split(","):
                                if ':' not in part:
                                    continue
                                k, v = part.split(':', 1)
                                k = k.strip().strip('"')
                                v = v.strip().strip('"')
                                if k:
                                    result[k] = v
                            data = result
                        except Exception:
                            logging.debug(f"Key Vault: secret '{secret_name}' could not be parsed as JSON")
                            continue
                    if not isinstance(data, dict):
                        logging.debug(f"Key Vault: secret '{secret_name}' value is not a JSON object")
                        continue
                    for attr, json_key in refs:
                        if json_key is not None:
                            val = data.get(json_key)
                            if attr in kv_secrets:
                                continue
                            if val is not None and val != "":
                                kv_secrets[attr] = str(val)
                else:
                    for attr, json_key in refs:
                        if json_key is None and raw_value:
                            kv_secrets[attr] = raw_value
                            break
            except Exception as exc:
                logging.debug(f"Key Vault: failed to fetch secret '{secret_name}': {exc}")
                continue
        cls._kv_secrets = kv_secrets
        return kv_secrets

    @classmethod
    def _validate_api_keys(cls):
        provider = getattr(cls, "MODEL_PROVIDER", "").lower()
        if provider == "openai":
            if not getattr(cls, "OPENAI_API_KEY", ""):
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider.")
        elif provider == "azure":
            if not getattr(cls, "AZURE_OPENAI_API_KEY", ""):
                raise ValueError("AZURE_OPENAI_API_KEY is required for Azure provider.")
        elif provider == "anthropic":
            if not getattr(cls, "ANTHROPIC_API_KEY", ""):
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider.")
        elif provider == "google":
            if not getattr(cls, "GOOGLE_API_KEY", ""):
                raise ValueError("GOOGLE_API_KEY is required for Google provider.")

    @classmethod
    def get_llm_kwargs(cls):
        kwargs = {}
        model_lower = (getattr(cls, "LLM_MODEL", "") or "").lower()
        if not any(model_lower.startswith(m) for m in cls._TEMPERATURE_UNSUPPORTED):
            kwargs["temperature"] = getattr(cls, "LLM_TEMPERATURE", None)
        if any(model_lower.startswith(m) for m in cls._MAX_TOKENS_UNSUPPORTED):
            kwargs["max_completion_tokens"] = getattr(cls, "LLM_MAX_TOKENS", None)
        else:
            kwargs["max_tokens"] = getattr(cls, "LLM_MAX_TOKENS", None)
        return kwargs

    @classmethod
    def validate(cls):
        cls._validate_api_keys()

def _initialize_config():
    # Load Key Vault config from .env
    USE_KEY_VAULT = os.getenv("USE_KEY_VAULT", "").lower() in ("true", "1", "yes")
    KEY_VAULT_URI = os.getenv("KEY_VAULT_URI", "")
    AZURE_USE_DEFAULT_CREDENTIAL = os.getenv("AZURE_USE_DEFAULT_CREDENTIAL", "").lower() in ("true", "1", "yes")

    setattr(Config, "USE_KEY_VAULT", USE_KEY_VAULT)
    setattr(Config, "KEY_VAULT_URI", KEY_VAULT_URI)
    setattr(Config, "AZURE_USE_DEFAULT_CREDENTIAL", AZURE_USE_DEFAULT_CREDENTIAL)

    # Load Key Vault secrets if enabled
    if USE_KEY_VAULT:
        Config._load_keyvault_secrets()

    # Azure AI Search variables (always from .env)
    AZURE_SEARCH_VARS = ["AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY", "AZURE_SEARCH_INDEX_NAME"]
    # Service Principal vars (skip if using DefaultAzureCredential)
    AZURE_SP_VARS = ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]

    # All config variables required by agent and observability
    CONFIG_VARIABLES = [
        # General
        "ENVIRONMENT",
        # Key Vault/Service Principal
        "AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET",
        # LLM / Model
        "MODEL_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS",
        "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        # API Requirements
        "AZURE_CONTENT_SAFETY_ENDPOINT", "AZURE_CONTENT_SAFETY_KEY",
        # Azure AI Search (RAG)
        "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY", "AZURE_SEARCH_INDEX_NAME",
        # Agent identity
        "AGENT_NAME", "AGENT_ID", "PROJECT_NAME", "PROJECT_ID", "SERVICE_NAME", "SERVICE_VERSION",
        # Observability DB
        "OBS_DATABASE_TYPE", "OBS_AZURE_SQL_SERVER", "OBS_AZURE_SQL_DATABASE", "OBS_AZURE_SQL_PORT",
        "OBS_AZURE_SQL_USERNAME", "OBS_AZURE_SQL_PASSWORD", "OBS_AZURE_SQL_SCHEMA", "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE",
        # Content Safety runtime flags
        "CONTENT_SAFETY_ENABLED", "CONTENT_SAFETY_SEVERITY_THRESHOLD",
        # LLM Models list (for observability cost tracking)
        "LLM_MODELS",
        # Validation config path (domain-specific)
        "VALIDATION_CONFIG_PATH",
        # Version
        "VERSION",
    ]

    for var_name in CONFIG_VARIABLES:
        # Skip Service Principal vars if using DefaultAzureCredential
        if var_name in AZURE_SP_VARS and AZURE_USE_DEFAULT_CREDENTIAL:
            continue

        value = None

        # Azure AI Search variables: always from .env
        if var_name in AZURE_SEARCH_VARS:
            value = os.getenv(var_name)
        # Key Vault priority for all others
        elif USE_KEY_VAULT and var_name in Config._kv_secrets:
            value = Config._kv_secrets[var_name]
        else:
            value = os.getenv(var_name)

        # Special: OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE defaults to "yes" if not found
        if var_name == "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE":
            if not value:
                value = "yes"
        # Special: LLM_MODELS is a JSON list (for observability cost tracking)
        elif var_name == "LLM_MODELS":
            if value:
                try:
                    import json
                    value = json.loads(value)
                except Exception:
                    logging.warning(f"Invalid JSON for {var_name}: {value}")
                    value = []
            else:
                value = []
        # Special: CONTENT_SAFETY_ENABLED is bool
        elif var_name == "CONTENT_SAFETY_ENABLED":
            if value is not None:
                value = str(value).lower() in ("true", "1", "yes", "on")
            else:
                value = False
        # Special: CONTENT_SAFETY_SEVERITY_THRESHOLD is int
        elif var_name == "CONTENT_SAFETY_SEVERITY_THRESHOLD":
            if value:
                try:
                    value = int(value)
                except ValueError:
                    logging.warning(f"Invalid integer value for {var_name}: {value}")
                    value = 2
            else:
                value = 2
        # Special: LLM_TEMPERATURE is float
        elif var_name == "LLM_TEMPERATURE":
            if value:
                try:
                    value = float(value)
                except ValueError:
                    logging.warning(f"Invalid float value for {var_name}: {value}")
                    value = None
            else:
                value = None
        # Special: LLM_MAX_TOKENS, OBS_AZURE_SQL_PORT are int
        elif var_name in ("LLM_MAX_TOKENS", "OBS_AZURE_SQL_PORT"):
            if value:
                try:
                    value = int(value)
                except ValueError:
                    logging.warning(f"Invalid integer value for {var_name}: {value}")
                    value = None
            else:
                value = None

        # Warn if missing (except for LLM_MODELS, which can be empty list)
        if (value is None or value == "" or (var_name == "LLM_MODELS" and value == [])) and var_name != "OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE":
            logging.warning(f"Configuration variable {var_name} not found in .env file")
            if var_name == "LLM_MODELS":
                value = []
            elif var_name in ("LLM_TEMPERATURE", "LLM_MAX_TOKENS", "OBS_AZURE_SQL_PORT", "CONTENT_SAFETY_SEVERITY_THRESHOLD"):
                value = None
            elif var_name == "CONTENT_SAFETY_ENABLED":
                value = False
            else:
                value = ""

        setattr(Config, var_name, value)

# Call at module level
_initialize_config()

# Settings instance (backward compatibility with observability module)
settings = Config()
