"""
Dynamic LLM Router

Detects which API keys are present in .env and routes inference
to the appropriate provider using a dictionary-based dispatch pattern.

Supported providers: huggingface, groq, gemini
"""
from dotenv import dotenv_values

env_vars = dotenv_values(".env")

# Maps provider name -> the .env key that must be set for it to be available
_PROVIDER_ENV_KEYS = {
    "huggingface": "HUGGINGFACE_TOKEN",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _build_huggingface(model_id: str):
    from langchain_community.llms import HuggingFaceEndpoint
    token = env_vars.get("HUGGINGFACE_TOKEN")
    return HuggingFaceEndpoint(repo_id=model_id, temperature=0.5, token=token)


def _build_groq(model_id: str):
    from langchain_groq import ChatGroq
    api_key = env_vars.get("GROQ_API_KEY")
    return ChatGroq(model_name=model_id, api_key=api_key, temperature=0.5)


def _build_gemini(model_id: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = env_vars.get("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model=model_id, google_api_key=api_key, temperature=0.5)


# Dict-based dispatch: maps provider name -> builder function
_PROVIDER_BUILDERS = {
    "huggingface": _build_huggingface,
    "groq": _build_groq,
    "gemini": _build_gemini,
}


def get_available_providers() -> list:
    """
    Return the list of providers that have API keys configured in .env.
    Used by the frontend to expose only available providers.
    """
    return [
        name for name, env_key in _PROVIDER_ENV_KEYS.items()
        if env_vars.get(env_key)
    ]


def build_llm(provider: str, model_id: str):
    """
    Build and return an LLM instance for the given provider and model ID.

    Args:
        provider: One of 'huggingface', 'groq', or 'gemini'.
        model_id: The provider-specific model identifier.

    Returns:
        A LangChain LLM or Chat model instance.

    Raises:
        ValueError: If the provider is unknown or its API key is not configured.
    """
    if provider not in _PROVIDER_BUILDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {list(_PROVIDER_BUILDERS.keys())}"
        )

    available = get_available_providers()
    if provider not in available:
        env_key = _PROVIDER_ENV_KEYS[provider]
        raise ValueError(
            f"Provider '{provider}' is not configured. "
            f"Set {env_key} in your .env file."
        )

    return _PROVIDER_BUILDERS[provider](model_id)
