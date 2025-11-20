import time
import threading
from collections import deque
from typing import Callable, Any
import logging
from functools import wraps
import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for LLM API calls."""
    
    def __init__(
        self,
        max_calls: int = config.LLM_RATE_LIMIT_CALLS,
        window_seconds: int = config.LLM_RATE_LIMIT_WINDOW
    ):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = deque()
        self.lock = threading.Lock()
        
        logger.info(
            f"Rate limiter initialized: {max_calls} calls per {window_seconds}s"
        )
    
    def _clean_old_calls(self):
        """Remove calls outside the current window."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        while self.calls and self.calls[0] < cutoff_time:
            self.calls.popleft()
    
    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire permission to make a call.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                self._clean_old_calls()
                
                if len(self.calls) < self.max_calls:
                    self.calls.append(time.time())
                    logger.debug(
                        f"Rate limit acquired ({len(self.calls)}/{self.max_calls})"
                    )
                    return True
                
                # Calculate wait time
                if self.calls:
                    oldest_call = self.calls[0]
                    wait_time = (oldest_call + self.window_seconds) - time.time()
                    
                    if wait_time > 0:
                        logger.warning(
                            f"Rate limit reached. Waiting {wait_time:.1f}s "
                            f"({len(self.calls)}/{self.max_calls} calls used)"
                        )
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.error("Rate limiter timeout exceeded")
                return False
            
            # Wait a bit before retrying
            time.sleep(min(1.0, max(0.1, wait_time if 'wait_time' in locals() else 1.0)))
    
    def get_current_usage(self) -> dict:
        """Get current rate limit usage."""
        with self.lock:
            self._clean_old_calls()
            return {
                "calls_used": len(self.calls),
                "calls_available": self.max_calls - len(self.calls),
                "window_seconds": self.window_seconds,
                "usage_percentage": (len(self.calls) / self.max_calls) * 100
            }


class CircuitBreaker:
    """
    Circuit breaker pattern for LLM calls.
    Opens circuit after consecutive failures to prevent cascading issues.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.lock = threading.Lock()
        
        logger.info(
            f"Circuit breaker initialized: threshold={failure_threshold}, "
            f"recovery={recovery_timeout}s"
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half_open"
                    logger.info("Circuit breaker: Attempting recovery (half-open)")
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit breaker is open. Too many failures. "
                        f"Retry after {self.recovery_timeout}s"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            if self.state == "half_open":
                self.state = "closed"
                logger.info("Circuit breaker: Recovered (closed)")
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    f"Circuit breaker: OPENED after {self.failure_count} failures"
                )
    
    def reset(self):
        """Manually reset the circuit breaker."""
        with self.lock:
            self.failure_count = 0
            self.state = "closed"
            self.last_failure_time = None
            logger.info("Circuit breaker: Manually reset")
    
    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time
        }


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


# Global instances
llm_rate_limiter = RateLimiter()
llm_circuit_breaker = CircuitBreaker()


def rate_limited(func: Callable) -> Callable:
    """Decorator to add rate limiting to a function."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Acquire rate limit token
        if not llm_rate_limiter.acquire(timeout=config.LLM_TIMEOUT):
            raise RateLimitExceeded("Rate limit timeout exceeded")
        
        # Execute with circuit breaker
        try:
            return llm_circuit_breaker.call(func, *args, **kwargs)
        except Exception as e:
            logger.error(f"Rate-limited call failed: {e}")
            raise
    
    return wrapper


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


class RetryWithBackoff:
    """Retry mechanism with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = config.LLM_MAX_RETRIES,
        base_delay: float = config.LLM_RETRY_DELAY,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for retry with exponential backoff."""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        logger.error(
                            f"Max retries ({self.max_retries}) exceeded for {func.__name__}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} for {func.__name__} "
                        f"after {delay:.1f}s delay. Error: {e}"
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper


# Decorator instances
retry_with_backoff = RetryWithBackoff()