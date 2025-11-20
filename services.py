import os
import docker
import mlflow
import logging

import config

logger = logging.getLogger(__name__)

# --- 1. Logging Service ---

def setup_logging():
    """Configures the root logger for the application."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logger.info("Logging configured.")

# --- 2. MLflow Service ---

def setup_mlflow_run(recipe: dict) -> str:
    """
    Sets up MLflow, creates an experiment if it doesn't exist,
    and starts a new run.
    
    Args:
        recipe: The experiment recipe to log as parameters.
        
    Returns:
        The run_id of the newly created run.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    
    experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        logger.info(f"MLflow: Creating new experiment '{config.MLFLOW_EXPERIMENT_NAME}'")
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    run = mlflow.start_run()
    run_id = run.info.run_id
    
    try:
        # Log parameters safely
        safe_params = {k: v for k, v in recipe.items() if isinstance(v, (str, int, float, bool))}
        mlflow.log_params(safe_params)
    except Exception:
        logger.warning("Could not log all recipe params to MLflow.")
        
    logger.info(f"MLflow: Started run with ID: {run_id}")
    return run_id

# --- 3. Docker Services ---

def safe_torch_install_line(cuda_version: str = "cpu") -> str:
    """Returns the correct pip install command for PyTorch (CPU or GPU)."""
    if cuda_version == "cpu":
        index_url = "https://download.pytorch.org/whl/cpu"
    else:
        index_url = f"https://download.pytorch.org/whl/{cuda_version}"
    return f"RUN pip install --no-cache-dir torch torchvision --index-url {index_url}"

def capture_image_build_logs(client, path, tag):
    """Builds a Docker image and streams logs."""
    try:
        img, logs_stream = client.images.build(path=path, tag=tag, rm=True, pull=True)
        log_txt = ""
        for chunk in logs_stream:
            if isinstance(chunk, dict) and 'stream' in chunk:
                line = chunk['stream'].strip()
                if line:
                    logger.info(f"BUILD: {line}")
                    log_txt += line + "\n"
            else:
                logger.debug(f"build chunk: {chunk}")
        return img, log_txt
    except docker.errors.BuildError as e:
        logger.error(f"Docker build ERROR: {e}")
        log_txt = "\n".join([chunk.get('stream', '') for chunk in getattr(e, "build_log", []) if isinstance(chunk, dict)])
        logger.error(f"BUILD LOGS:\n{log_txt}")
        return None, log_txt
    except Exception as e:
        logger.error(f"Docker build UNEXPECTED ERROR: {e}", exc_info=True)
        return None, str(e)

def run_container_and_collect_logs(client, tag, mlflow_run_id):
    """Runs the container and returns (success, logs)."""
    try:
        host_db_path = os.path.abspath(str(config.MLFLOW_TRACKING_URI).replace("sqlite:///", ""))
        container_db_path = "/app/mlflow.db"

        if not os.path.exists(host_db_path):
            try:
                open(host_db_path, 'a').close()
                logger.info(f"Created empty mlflow.db at {host_db_path}")
            except Exception as e:
                logger.error(f"Could not create empty mlflow.db: {e}")
                return False, f"Could not create mlflow.db on host: {e}"

        volumes_map = {
            host_db_path: {
                'bind': container_db_path,
                'mode': 'rw'
            }
        }

        container = client.containers.run(
            tag,
            detach=True,
            remove=True,
            environment={"MLFLOW_RUN_ID": mlflow_run_id},
            volumes=volumes_map
        )

        logs_text_lines = []
        for chunk in container.logs(stream=True):
            if isinstance(chunk, bytes):
                try:
                    line = chunk.decode(errors="replace").rstrip("\n")
                except Exception:
                    line = str(chunk)
            else:
                line = str(chunk).rstrip("\n")

            print(line)
            logger.info(line)
            logs_text_lines.append(line)

        txt = "\n".join(logs_text_lines)

        if "Traceback (most recent call last):" in txt or "\nError:" in txt or "SyntaxError" in txt:
            logger.error("Docker run FAILED with Python error.")
            return (False, txt)

        logger.info("Docker run SUCCESS.")
        return (True, txt)

    except docker.errors.ContainerError as e:
        stderr = e.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        logger.error(f"Docker run CONTAINER ERROR: {stderr}")
        return False, stderr
    except Exception as e:
        logger.error(f"Docker run UNEXPECTED ERROR: {e}", exc_info=True)
        return False, str(e)