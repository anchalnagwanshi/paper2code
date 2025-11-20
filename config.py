import logging
from pathlib import Path

# --- Directories ---
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
BUILD_DIR = BASE_DIR / "build_context"
METRICS_DIR = BASE_DIR / "metrics"
CACHE_DIR = BASE_DIR / ".cache"
TEST_DIR = BASE_DIR / "tests"

# Create directories
for dir_path in [LOG_DIR, BUILD_DIR, METRICS_DIR, CACHE_DIR, TEST_DIR]:
    dir_path.mkdir(exist_ok=True)

# --- Logging ---
LOG_LEVEL = logging.INFO
LOG_FILE = LOG_DIR / "paper2code.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"
METRICS_LOG_FILE = LOG_DIR / "metrics.log"
LOG_ROTATION_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# --- MLflow ---
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "paper2code"

# --- LLM ---
LLM_MODEL = "phi3"
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT = 300  # 5 minutes
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 2  # seconds

# Rate Limiting
LLM_RATE_LIMIT_CALLS = 100  # calls per window
LLM_RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
LLM_COST_PER_CALL = 0.0  # Set based on your provider

# --- Docker ---
DOCKER_IMAGE_TAG = "paper2code:latest"
BUILD_CONTEXT_DIR = str(BUILD_DIR)
DOCKER_BUILD_TIMEOUT = 600  # 10 minutes
DOCKER_RUN_TIMEOUT = 1800  # 30 minutes
CUDA_VERSION = "cpu"  # or "cu121" for CUDA 12.1

# --- Graph ---
MAX_RETRIES = 3
RECURSION_LIMIT = 20
ENABLE_CHECKPOINTING = True
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# --- Validation ---
ENABLE_PROGRESSIVE_VALIDATION = True
VALIDATION_STAGES = ["imports", "data_load", "model_init", "forward", "backward", "training"]
VALIDATION_TIMEOUT = 300  # 5 minutes per stage

# --- Error Recovery ---
ENABLE_INTELLIGENT_DEBUGGING = True
ENABLE_CODE_ANALYSIS = True
MAX_ERROR_CONTEXT_LINES = 50

# --- Paper Processing ---
MAX_PAPER_LENGTH = 100000  # characters
FOCUS_SECTIONS = ["abstract", "methodology", "method", "experiments", "implementation", "training"]
EXTRACT_FIGURES = False  # Advanced feature
EXTRACT_TABLES = False   # Advanced feature

# --- Code Generation ---
CODE_GENERATION_STRATEGY = "multi_stage"  # "single_pass" or "multi_stage"
ENABLE_CODE_TEMPLATES = True
VALIDATE_BEFORE_EXECUTION = True
MIN_CODE_LENGTH = 500  # characters

# --- Caching ---
ENABLE_LLM_CACHE = True
CACHE_TTL = 86400  # 24 hours
MAX_CACHE_SIZE = 1000  # entries

# --- Monitoring ---
ENABLE_METRICS = True
METRICS_EXPORT_INTERVAL = 60  # seconds
ENABLE_PROMETHEUS = False  # Set True if using Prometheus

# --- User Feedback ---
ENABLE_FEEDBACK_LOOP = True
FEEDBACK_FILE = METRICS_DIR / "user_feedback.jsonl"

# --- Testing ---
TEST_PAPERS_DIR = TEST_DIR / "papers"
TEST_PAPERS_DIR.mkdir(exist_ok=True)
ENABLE_AUTO_TESTING = False

# --- Feature Flags ---
FEATURES = {
    "enhanced_researcher": True,
    "intelligent_debugger": True,
    "progressive_validation": True,
    "code_analysis": True,
    "human_review": False,
    "parallel_processing": False,
}

# --- Timeouts ---
TIMEOUTS = {
    "researcher": 300,
    "coder": 300,
    "qa": 1800,
    "total_pipeline": 3600,
}

# --- Cost Tracking ---
ENABLE_COST_TRACKING = True
COST_BUDGET_PER_RUN = 1.0  # dollars
COST_ALERT_THRESHOLD = 0.8  # 80% of budget