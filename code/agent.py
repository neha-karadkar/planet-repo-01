import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

from config import Config

# =========================
# CONSTANTS
# =========================

SYSTEM_PROMPT = (
    "You are an expert planetary science analyst. Your task is to deliver a comprehensive, professional comparative analysis of Earth and Jupiter using only information retrieved from Earth.pdf and Jupiter.pdf. For each planet, explicitly cite their equatorial diameter in both miles and kilometers, and describe the scale difference (including how many Earths could fit inside Jupiter). Compare their average distances from the Sun, explaining the significance of their orbital positions. Format your response clearly, using bullet points or tables where appropriate. If any required measurement or comparison is missing from the retrieved content, politely inform the user and suggest consulting authoritative scientific sources."
)
OUTPUT_FORMAT = (
    "- Structured summary with clear headings for each planet\n"
    "- Explicit citation of diameter and distance values (miles and kilometers)\n"
    "- Comparative scale explanation (e.g., number of Earths fitting inside Jupiter)\n"
    "- Orbital position comparison\n"
    "- Use bullet points or tables for clarity"
)
FALLBACK_RESPONSE = (
    "The requested measurements or comparisons are not available in the provided knowledge base documents. Please consult additional scientific resources for further information."
)
SELECTED_DOCUMENT_TITLES = ["Earth.pdf", "Jupiter.pdf"]
TOP_K = 5

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# OBSERVABILITY LIFESPAN
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="Planetary Comparative Analysis Agent",
    description="Compares Earth and Jupiter using only knowledge base documents, with explicit measurement citation and structured output.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

# =========================
# Pydantic Models
# =========================

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was successful")
    result: Optional[str] = Field(None, description="The formatted comparative analysis or fallback message")
    error: Optional[str] = Field(None, description="Error message if any")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input or retrying")


# =========================
# Utility: LLM Output Sanitizer
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# Error Handler
# =========================

class ErrorHandler:
    """Centralized error handler for the agent."""

    ERROR_MAP = {
        "DOC_NOT_FOUND": FALLBACK_RESPONSE,
        "MEASUREMENT_MISSING": FALLBACK_RESPONSE,
        "RETRIEVAL_ERROR": "An error occurred while retrieving planetary data. Please try again later.",
        "LLM_ERROR": "An error occurred while generating the comparative analysis. Please try again later.",
        "VALIDATION_ERROR": "Invalid input. Please check your request and try again.",
        "UNKNOWN": "An unknown error occurred. Please try again later."
    }

    @staticmethod
    def handle_error(error_code: str, context: Optional[Any] = None) -> str:
        """Map error code to fallback or error response."""
        logging.getLogger("agent").info(f"Error handled: {error_code} | Context: {context}")
        return ErrorHandler.ERROR_MAP.get(error_code, FALLBACK_RESPONSE)

# =========================
# Chunk Retriever
# =========================

class ChunkRetriever:
    """Retrieves relevant chunks from Azure AI Search, filtered by document titles."""

    def __init__(self):
        self._search_client = None

    def _get_search_client(self):
        if self._search_client is None:
            endpoint = Config.AZURE_SEARCH_ENDPOINT
            api_key = Config.AZURE_SEARCH_API_KEY
            index_name = Config.AZURE_SEARCH_INDEX_NAME
            if not endpoint or not api_key or not index_name:
                raise ValueError("Azure AI Search credentials are not configured.")
            self._search_client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )
        return self._search_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, document_titles: List[str], top_k: int = 5) -> List[str]:
        """Retrieve top-K relevant chunks from Azure AI Search, filtered by document titles."""
        search_client = self._get_search_client()
        openai_client = openai.AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )

        # Step 1: Embed the query (system prompt)
        _t0 = _time.time()
        embedding_resp = await openai_client.embeddings.create(
            input=query,
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        )
        try:
            trace_tool_call(
                tool_name="openai_client.embeddings.create",
                latency_ms=int((_time.time() - _t0) * 1000),
                args={"input": query, "model": Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"},
                output=str(embedding_resp)[:200],
                status="success"
            )
        except Exception:
            pass

        vector_query = VectorizedQuery(
            vector=embedding_resp.data[0].embedding,
            k_nearest_neighbors=top_k,
            fields="vector"
        )

        # Step 2: Build OData filter for selected document titles
        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": ["chunk", "title"],
        }
        if document_titles:
            odata_parts = [f"title eq '{t}'" for t in document_titles]
            search_kwargs["filter"] = " or ".join(odata_parts)

        # Step 3: Search
        _t1 = _time.time()
        try:
            results = search_client.search(**search_kwargs)
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t1) * 1000),
                    args=search_kwargs,
                    output="retrieved",
                    status="success"
                )
            except Exception:
                pass
            context_chunks = [r["chunk"] for r in results if r.get("chunk")]
            return context_chunks
        except Exception as e:
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t1) * 1000),
                    args=search_kwargs,
                    output=str(e),
                    status="error",
                    error=e
                )
            except Exception:
                pass
            raise

# =========================
# LLM Service
# =========================

class LLMService:
    """Handles LLM calls to Azure OpenAI."""

    def __init__(self):
        self._client = None

    def _get_llm_client(self):
        api_key = Config.AZURE_OPENAI_API_KEY
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not configured")
        return openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def generate_response(self, prompt: str, user_query: str, context_chunks: List[str]) -> str:
        """Call LLM with system prompt, user query, and context chunks; return generated text."""
        client = self._get_llm_client()
        context_text = "\n\n".join(context_chunks) if context_chunks else ""
        system_message = prompt + "\n\nOutput Format: " + OUTPUT_FORMAT

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": context_text}
        ]

        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        try:
            response = await client.chat.completions.create(
                model=Config.LLM_MODEL or "gpt-4.1",
                messages=messages,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return content
        except Exception as e:
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    status="error",
                    error=e,
                    response_summary=str(e)[:200]
                )
            except Exception:
                pass
            raise

# =========================
# Response Formatter
# =========================

class ResponseFormatter:
    """Ensures output matches required structure and clarity."""

    @staticmethod
    def format_response(raw_llm_output: str) -> str:
        """Format LLM output into structured summary with headings, bullet points, tables, and explicit citations."""
        # The LLM is responsible for structure, but we sanitize and ensure fallback formatting.
        text = sanitize_llm_output(raw_llm_output, content_type="text")
        if not text or len(text.strip()) == 0:
            return FALLBACK_RESPONSE
        return text

# =========================
# Main Agent
# =========================

class PlanetaryComparativeAnalysisAgent:
    """Orchestrates planetary comparison analysis."""

    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.response_formatter = ResponseFormatter()
        self.error_handler = ErrorHandler()
        self.guardrails_config = GUARDRAILS_CONFIG

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def analyze_planetary_comparison(self) -> Dict[str, Any]:
        """
        Main entry point; orchestrates retrieval, LLM call, formatting, and error handling.
        Returns:
            dict: {success, result, error, tips}
        """
        async with trace_step(
            "retrieve_chunks",
            step_type="tool_call",
            decision_summary="Retrieve relevant planetary chunks from Azure AI Search",
            output_fn=lambda r: f"{len(r)} chunks" if isinstance(r, list) else "0 chunks"
        ) as step:
            try:
                chunks = await self.chunk_retriever.retrieve_chunks(
                    query=SYSTEM_PROMPT,
                    document_titles=SELECTED_DOCUMENT_TITLES,
                    top_k=TOP_K
                )
                step.capture(chunks)
            except Exception as e:
                error_msg = self.error_handler.handle_error("RETRIEVAL_ERROR", str(e))
                return {
                    "success": False,
                    "result": FALLBACK_RESPONSE,
                    "error": error_msg,
                    "tips": "Ensure the knowledge base contains the required planetary documents."
                }

        if not chunks or len(chunks) == 0:
            error_msg = self.error_handler.handle_error("DOC_NOT_FOUND")
            return {
                "success": False,
                "result": FALLBACK_RESPONSE,
                "error": error_msg,
                "tips": "Ensure Earth.pdf and Jupiter.pdf are present in the knowledge base."
            }

        async with trace_step(
            "llm_generate_response",
            step_type="llm_call",
            decision_summary="Generate comparative analysis using LLM",
            output_fn=lambda r: f"{len(r)} chars" if isinstance(r, str) else "no output"
        ) as step:
            try:
                raw_llm_output = await self.llm_service.generate_response(
                    prompt=SYSTEM_PROMPT,
                    user_query=SYSTEM_PROMPT,
                    context_chunks=chunks
                )
                step.capture(raw_llm_output)
            except Exception as e:
                error_msg = self.error_handler.handle_error("LLM_ERROR", str(e))
                return {
                    "success": False,
                    "result": FALLBACK_RESPONSE,
                    "error": error_msg,
                    "tips": "Try again later or check LLM configuration."
                }

        async with trace_step(
            "format_response",
            step_type="format",
            decision_summary="Format LLM output for clarity and citation",
            output_fn=lambda r: f"{len(r)} chars" if isinstance(r, str) else "no output"
        ) as step:
            try:
                formatted = self.response_formatter.format_response(raw_llm_output)
                step.capture(formatted)
            except Exception as e:
                error_msg = self.error_handler.handle_error("UNKNOWN", str(e))
                return {
                    "success": False,
                    "result": FALLBACK_RESPONSE,
                    "error": error_msg,
                    "tips": "Formatting failed. Please try again."
                }

        return {
            "success": True,
            "result": formatted,
            "error": None,
            "tips": None
        }

# =========================
# FastAPI Endpoints
# =========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint():
    """
    Main endpoint for planetary comparative analysis.
    No user input required; SYSTEM_PROMPT and document filter are internal.
    """
    agent = PlanetaryComparativeAnalysisAgent()
    try:
        result = await agent.analyze_planetary_comparison()
        return QueryResponse(**result)
    except Exception as e:
        logging.getLogger("agent").exception("Unhandled error in /query endpoint")
        return QueryResponse(
            success=False,
            result=FALLBACK_RESPONSE,
            error=str(e),
            tips="An unexpected error occurred. Please try again later."
        )

# =========================
# FastAPI Exception Handlers
# =========================

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "result": None,
            "error": "Malformed JSON or invalid request.",
            "tips": "Check your JSON formatting (quotes, commas, brackets) and ensure all required fields are present."
        },
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "result": None,
            "error": "Malformed JSON or invalid request.",
            "tips": "Check your JSON formatting (quotes, commas, brackets) and ensure all required fields are present."
        },
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.getLogger("agent").exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "result": None,
            "error": "Internal server error.",
            "tips": "Try again later or contact support."
        },
    )

# =========================
# Entrypoint
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())