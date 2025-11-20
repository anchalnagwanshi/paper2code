import os
import json
import re
import logging
from contextlib import contextmanager
from typing import Dict, Optional
import docker
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import config
from schema import GraphState, ExperimentRecipe
from logging_config import ContextLogger, error_tracker, metrics_logger
from monitoring import metrics_collector, cost_tracker, performance_monitor
from rate_limiter import rate_limited, retry_with_backoff
from cache import cached_llm_call, llm_cache
import prompts
import templates
import services
import threading

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


@contextmanager
def timeout(seconds: int, error_message: str = "Operation timed out"):
    """Context manager for timeout handling (Windows compatible)."""
    timer = None
    
    def timeout_handler():
        raise TimeoutError(error_message)
    
    try:
        # Use threading.Timer for Windows compatibility
        timer = threading.Timer(seconds, timeout_handler)
        timer.start()
        yield
    finally:
        if timer:
            timer.cancel()



# --- 1. Enhanced Researcher Node ---

def researcher_node(state: GraphState) -> GraphState:
    """
    Reads paper and extracts experiment recipe with comprehensive error handling.
    """
    logger.info("=" * 80)
    logger.info("RESEARCHER NODE")
    logger.info("=" * 80)
    
    context = {"operation": "researcher", "paper_path": state.get("paper_path")}
    
    with ContextLogger(logger, context):
        performance_monitor.checkpoint("researcher_start")
        
        try:
            # Validate input
            paper_path = state.get("paper_path")
            if not paper_path:
                raise ValueError("paper_path not provided in state")
            
            if not os.path.exists(paper_path):
                raise FileNotFoundError(f"Paper file not found: {paper_path}")
            
            # Load paper with timeout
            logger.info(f"Loading paper: {paper_path}")
            with timeout(config.TIMEOUTS["researcher"], "Paper loading timed out"):
                paper_text = _load_paper_with_validation(paper_path)
            
            logger.info(f"Paper loaded: {len(paper_text)} characters")
            metrics_logger.log_metric("paper_length", len(paper_text))
            
            # Extract recipe with LLM
            logger.info("Extracting experiment recipe...")
            recipe = _extract_recipe_with_retry(paper_text)
            
            # Validate recipe
            _validate_recipe(recipe)
            
            recipe_dict = recipe.dict()
            logger.info("Recipe extracted successfully")
            logger.info(json.dumps(recipe_dict, indent=2))
            
            # Record metrics
            duration = performance_monitor.get_duration("researcher_start")
            metrics_collector.record_stage_duration("researcher", duration)
            metrics_logger.log_event("researcher_success", {
                "paper_path": paper_path,
                "duration": duration,
                "paper_length": len(paper_text)
            })
            
            return {
                **state,
                "recipe": recipe_dict,
                "error": None,
                "retries": 0
            }
        
        except TimeoutError as e:
            logger.error(f"Researcher timeout: {e}")
            error_tracker.log_error(e, context, "timeout", "error")
            return {**state, "error": f"Researcher timeout: {e}"}
        
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            error_tracker.log_error(e, context, "file_not_found", "error")
            return {**state, "error": str(e)}
        
        except Exception as e:
            logger.error(f"Researcher error: {e}", exc_info=True)
            error_tracker.log_error(e, context, "researcher_error", "critical")
            return {**state, "error": f"Researcher failed: {e}"}


def _load_paper_with_validation(paper_path: str) -> str:
    """Load paper and validate content."""
    try:
        if paper_path.endswith(".pdf"):
            loader = PyPDFLoader(paper_path)
        elif paper_path.endswith(".txt"):
            loader = TextLoader(paper_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {paper_path}")
        
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        
        # Validate content
        if not text or len(text) < 100:
            raise ValueError("Paper content too short or empty")
        
        if len(text) > config.MAX_PAPER_LENGTH:
            logger.warning(
                f"Paper very long ({len(text)} chars), truncating to {config.MAX_PAPER_LENGTH}"
            )
            text = text[:config.MAX_PAPER_LENGTH]
        
        return text
    
    except Exception as e:
        logger.error(f"Failed to load paper: {e}")
        raise


@rate_limited
@retry_with_backoff
def _extract_recipe_with_retry(paper_text: str, model: str = config.LLM_MODEL) -> ExperimentRecipe:
    """Extract recipe using LLM with rate limiting and caching."""
    try:
        llm = ChatOllama(
            model=model,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT
        )
        structured_llm = llm.with_structured_output(ExperimentRecipe)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.RESEARCHER_SYSTEM_PROMPT),
            ("human", prompts.RESEARCHER_HUMAN_PROMPT)
        ])
        
        chain = prompt | structured_llm
        
        # Record LLM call
        metrics_collector.record_llm_call(
            tokens=len(paper_text) // 4,  # Rough estimate
            cost=config.LLM_COST_PER_CALL
        )
        cost_tracker.add_cost(config.LLM_COST_PER_CALL, "researcher")
        
        recipe = chain.invoke({"paper_text": paper_text})
        return recipe
    
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        raise


def _validate_recipe(recipe: ExperimentRecipe):
    """Validate extracted recipe for completeness."""
    issues = []
    
    if recipe.learning_rate <= 0:
        issues.append(f"Invalid learning rate: {recipe.learning_rate}")
    
    if recipe.batch_size <= 0:
        issues.append(f"Invalid batch size: {recipe.batch_size}")
    
    if "not specified" in recipe.model_architecture.lower():
        issues.append("Model architecture not specified")
    
    if "not specified" in recipe.dataset.lower():
        issues.append("Dataset not specified")
    
    if issues:
        logger.warning(f"Recipe validation issues: {issues}")
        metrics_logger.log_event("recipe_validation_warnings", {"issues": issues})


# --- 2. MLflow Tracking Node ---

def setup_tracking_node(state: GraphState) -> GraphState:
    """Sets up MLflow tracking with error handling."""
    logger.info("=" * 80)
    logger.info("MLFLOW SETUP NODE")
    logger.info("=" * 80)
    
    context = {"operation": "mlflow_setup"}
    
    with ContextLogger(logger, context):
        try:
            recipe = state["recipe"]
            
            logger.info("Initializing MLflow tracking...")
            run_id = services.setup_mlflow_run(recipe)
            
            logger.info(f"MLflow run started: {run_id}")
            metrics_logger.log_event("mlflow_setup_success", {"run_id": run_id})
            
            return {
                **state,
                "mlflow_run_id": run_id,
                "error": None
            }
        
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}", exc_info=True)
            error_tracker.log_error(e, context, "mlflow_error", "error")
            return {**state, "error": f"MLflow setup failed: {e}"}


# --- 3. Enhanced Coder Node ---

def coder_node(state: GraphState) -> GraphState:
    """
    Generates code with comprehensive error handling and validation.
    """
    logger.info("=" * 80)
    logger.info("CODER NODE")
    logger.info("=" * 80)
    
    retries = state.get("retries", 0)
    context = {
        "operation": "coder",
        "attempt": retries + 1,
        "max_retries": config.MAX_RETRIES
    }
    
    with ContextLogger(logger, context):
        performance_monitor.checkpoint("coder_start")
        
        try:
            recipe = state["recipe"]
            run_id = state["mlflow_run_id"]
            
            # Prepare fix text if retrying
            fix_text = ""
            if state.get("error"):
                fix_text = _prepare_fix_text(state)
                logger.warning(f"Retry attempt {retries + 1}/{config.MAX_RETRIES}")
                metrics_collector.record_retry("coder")
            
            # Generate code with timeout
            logger.info("Generating code...")
            with timeout(config.TIMEOUTS["coder"], "Code generation timed out"):
                code = _generate_code_with_fallback(recipe, run_id, fix_text)
            
            # Validate code
            is_valid, validation_errors = _validate_code_thoroughly(code)
            metrics_collector.record_code_quality(code, is_valid)
            
            if not is_valid:
                logger.warning(f"Code validation issues: {validation_errors}")
                metrics_logger.log_event("code_validation_warnings", {
                    "errors": validation_errors
                })
            
            logger.info(f"Code generated: {len(code)} characters")
            
            # Record metrics
            duration = performance_monitor.get_duration("coder_start")
            metrics_collector.record_stage_duration("coder", duration)
            
            return {
                **state,
                "generated_code": code,
                "error": None,
                "retries": retries + 1
            }
        
        except TimeoutError as e:
            logger.error(f"Coder timeout: {e}")
            error_tracker.log_error(e, context, "timeout", "error")
            # Use fallback
            fallback_code = templates.fallback_train_py()
            return {
                **state,
                "generated_code": fallback_code,
                "error": None,
                "retries": retries + 1
            }
        
        except Exception as e:
            logger.error(f"Coder error: {e}", exc_info=True)
            error_tracker.log_error(e, context, "coder_error", "error")
            # Use fallback
            fallback_code = templates.fallback_train_py()
            return {
                **state,
                "generated_code": fallback_code,
                "error": str(e),
                "retries": retries + 1
            }


def _prepare_fix_text(state: GraphState) -> str:
    """Prepare helpful fix text from previous error."""
    error = state.get("error", "Unknown error")
    logs = state.get("docker_logs", "")
    
    # Extract relevant error lines
    error_lines = []
    if logs:
        lines = logs.split("\n")
        for i, line in enumerate(lines):
            if "error" in line.lower() or "exception" in line.lower():
                # Get context around error
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                error_lines = lines[start:end]
                break
    
    fix_text = f"""
PREVIOUS ATTEMPT FAILED. Please fix the code.

Error: {error}

Relevant logs:
{chr(10).join(error_lines[:config.MAX_ERROR_CONTEXT_LINES])}

Common fixes:
- Add missing imports
- Fix tensor dimensions
- Handle data loading errors
- Use correct API versions

Generate CORRECTED code now (Python only, no markdown):
"""
    return fix_text


@rate_limited
@retry_with_backoff
def _generate_code_with_fallback(recipe: dict, run_id: str, fix_text: str) -> str:
    """Generate code with LLM, fallback to template if needed."""
    try:
        llm = ChatOllama(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT
        )
        
        prompt = ChatPromptTemplate.from_template(prompts.CODER_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        
        # Record LLM call
        metrics_collector.record_llm_call(
            tokens=len(json.dumps(recipe)) // 4,
            cost=config.LLM_COST_PER_CALL
        )
        cost_tracker.add_cost(config.LLM_COST_PER_CALL, "coder")
        
        code = chain.invoke({
            "run_id": run_id,
            "recipe_json": json.dumps(recipe),
            "recipe": json.dumps(recipe),
            "fix_text": fix_text
        })
        
        # Clean code
        code = _clean_llm_output(code)
        
        # Validate
        if len(code) < config.MIN_CODE_LENGTH:
            logger.warning("LLM code too short, using template")
            return templates.render_train_py(recipe, run_id)
        
        # Try to compile
        try:
            compile(code, "<generated>", "exec")
            logger.info("âœ… LLM code validated")
            return code
        except SyntaxError as e:
            logger.warning(f"LLM code has syntax error: {e}, using template")
            return templates.render_train_py(recipe, run_id)
    
    except Exception as e:
        logger.error(f"Code generation failed: {e}, using fallback")
        return templates.fallback_train_py()


def _clean_llm_output(code: str) -> str:
    """Clean LLM output to extract pure Python code."""
    # Remove markdown
    code = code.replace("```python", "").replace("```", "").strip()
    
    # Remove bold text
    code = re.sub(r"\*\*.*?\*\*", "", code)
    
    # Remove control characters
    code = re.sub(r"[\x00-\x1f]", "", code)
    
    # Find actual code start (first import)
    lines = code.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            start_idx = i
            break
    
    code = "\n".join(lines[start_idx:])
    
    return code.strip()


def _validate_code_thoroughly(code: str) -> tuple:
    """Thoroughly validate generated code."""
    errors = []
    
    # Check length
    if len(code) < config.MIN_CODE_LENGTH:
        errors.append(f"Code too short: {len(code)} chars")
    
    # Check for required imports
    required_patterns = ["import torch", "import mlflow"]
    for pattern in required_patterns:
        if pattern not in code:
            errors.append(f"Missing required import: {pattern}")
    
    # Check for key functionality
    if "mlflow.log_metric" not in code and "mlflow.log_metrics" not in code:
        errors.append("Missing MLflow metric logging")
    
    if ".backward()" not in code:
        errors.append("Missing backward pass")
    
    if "optimizer.step()" not in code:
        errors.append("Missing optimizer step")
    
    # Try to compile
    try:
        compile(code, "<validation>", "exec")
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
    
    return len(errors) == 0, errors


# --- 4. Enhanced QA Node ---

def qa_node(state: GraphState) -> GraphState:
    """
    Test generated code in Docker sandbox with comprehensive error handling.
    """
    logger.info("=" * 80)
    logger.info("QA NODE")
    logger.info("=" * 80)
    
    context = {"operation": "qa"}
    
    with ContextLogger(logger, context):
        performance_monitor.checkpoint("qa_start")
        
        try:
            code = state["generated_code"]
            recipe = state["recipe"]
            run_id = state["mlflow_run_id"]
            
            # Prepare build context
            logger.info("Preparing build context...")
            _prepare_build_context(code, recipe)
            
            # Build Docker image
            logger.info("Building Docker image...")
            with timeout(config.DOCKER_BUILD_TIMEOUT, "Docker build timed out"):
                success, logs = _build_and_run_docker(run_id)
            
            # Record metrics
            duration = performance_monitor.get_duration("qa_start")
            metrics_collector.record_stage_duration("qa", duration)
            
            if not success:
                logger.error("Docker execution failed")
                metrics_logger.log_event("qa_failure", {
                    "duration": duration,
                    "logs_length": len(logs)
                })
                return {
                    **state,
                    "error": "Docker execution failed",
                    "docker_logs": logs
                }
            
            logger.info("âœ… QA passed successfully")
            metrics_logger.log_event("qa_success", {"duration": duration})
            
            return {
                **state,
                "docker_logs": logs,
                "error": None
            }
        
        except TimeoutError as e:
            logger.error(f"QA timeout: {e}")
            error_tracker.log_error(e, context, "timeout", "error")
            return {**state, "error": f"QA timeout: {e}"}
        
        except Exception as e:
            logger.error(f"QA error: {e}", exc_info=True)
            error_tracker.log_error(e, context, "qa_error", "error")
            return {**state, "error": str(e)}


def _prepare_build_context(code: str, recipe: dict):
    """Prepare Docker build context."""
    build_dir = config.BUILD_CONTEXT_DIR
    os.makedirs(build_dir, exist_ok=True)
    
    # Write recipe
    with open(os.path.join(build_dir, "recipe.json"), "w") as f:
        json.dump(recipe, f, indent=2)
    
    # Write code
    with open(os.path.join(build_dir, "train.py"), "w") as f:
        f.write(code)
    
    # Write Dockerfile
    torch_install = services.safe_torch_install_line(config.CUDA_VERSION)
    pip_install = "RUN pip install --no-cache-dir mlflow transformers datasets scikit-learn pandas"
    dockerfile = templates.get_dockerfile_template(torch_install, pip_install)
    
    with open(os.path.join(build_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile)


def _build_and_run_docker(run_id: str) -> tuple:
    """Build and run Docker container."""
    try:
        client = docker.from_env()
        
        # Build image
        logger.info(f"Building Docker image: {config.DOCKER_IMAGE_TAG}")
        image, build_logs = services.capture_image_build_logs(
            client,
            path=config.BUILD_CONTEXT_DIR,
            tag=config.DOCKER_IMAGE_TAG
        )
        
        if image is None:
            logger.error("Docker build failed")
            return False, build_logs
        
        logger.info("âœ… Docker build successful")
        
        # Run container
        logger.info("Running Docker container...")
        success, run_logs = services.run_container_and_collect_logs(
            client,
            config.DOCKER_IMAGE_TAG,
            run_id
        )
        
        all_logs = f"=== BUILD LOGS ===\n{build_logs}\n\n=== RUN LOGS ===\n{run_logs}"
        
        return success, all_logs
    
    except Exception as e:
        logger.error(f"Docker operation failed: {e}")
        return False, str(e)