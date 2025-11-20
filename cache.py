import hashlib
import json
import time
import pickle
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps
import logging
import config

logger = logging.getLogger(__name__)


class LLMCache:
    """Cache for LLM responses to avoid redundant API calls."""
    
    def __init__(
        self,
        cache_dir: Path = config.CACHE_DIR,
        ttl: int = config.CACHE_TTL,
        max_size: int = config.MAX_CACHE_SIZE
    ):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.max_size = max_size
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for fast access
        self.memory_cache = {}
        self.cache_metadata = self._load_metadata()
        
        # Cleanup old entries
        self._cleanup_expired()
        
        logger.info(
            f"LLM cache initialized: {len(self.cache_metadata)} entries, "
            f"TTL={ttl}s, max_size={max_size}"
        )
    
    def _generate_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key from prompt and parameters."""
        # Create a hashable representation
        cache_input = {
            "prompt": prompt,
            "model": model,
            **kwargs
        }
        
        # Sort dict for consistent hashing
        cache_str = json.dumps(cache_input, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, **kwargs) -> Optional[Any]:
        """Retrieve cached response if available and not expired."""
        if not config.ENABLE_LLM_CACHE:
            return None
        
        cache_key = self._generate_key(prompt, model, **kwargs)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if not self._is_expired(entry):
                logger.debug(f"Cache HIT (memory): {cache_key[:16]}...")
                return entry["response"]
            else:
                del self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                if not self._is_expired(entry):
                    logger.debug(f"Cache HIT (disk): {cache_key[:16]}...")
                    # Load into memory cache
                    self.memory_cache[cache_key] = entry
                    return entry["response"]
                else:
                    # Remove expired entry
                    cache_file.unlink()
                    if cache_key in self.cache_metadata:
                        del self.cache_metadata[cache_key]
            except Exception as e:
                logger.warning(f"Failed to load cache entry: {e}")
        
        logger.debug(f"Cache MISS: {cache_key[:16]}...")
        return None
    
    def set(self, prompt: str, model: str, response: Any, **kwargs):
        """Store response in cache."""
        if not config.ENABLE_LLM_CACHE:
            return
        
        cache_key = self._generate_key(prompt, model, **kwargs)
        
        entry = {
            "response": response,
            "timestamp": time.time(),
            "prompt_preview": prompt[:100],
            "model": model
        }
        
        # Store in memory cache
        self.memory_cache[cache_key] = entry
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
            
            # Update metadata
            self.cache_metadata[cache_key] = {
                "timestamp": entry["timestamp"],
                "size": cache_file.stat().st_size
            }
            self._save_metadata()
            
            logger.debug(f"Cache SET: {cache_key[:16]}...")
            
            # Check size limit
            if len(self.cache_metadata) > self.max_size:
                self._evict_oldest()
        
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def _is_expired(self, entry: dict) -> bool:
        """Check if cache entry has expired."""
        if self.ttl <= 0:
            return False
        age = time.time() - entry["timestamp"]
        return age > self.ttl
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        expired_keys = []
        
        for cache_key, metadata in self.cache_metadata.items():
            age = time.time() - metadata["timestamp"]
            if age > self.ttl:
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            del self.cache_metadata[cache_key]
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            self._save_metadata()
    
    def _evict_oldest(self):
        """Evict oldest cache entries when max size is exceeded."""
        # Sort by timestamp
        sorted_entries = sorted(
            self.cache_metadata.items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        # Remove oldest 10%
        num_to_remove = max(1, int(self.max_size * 0.1))
        
        for cache_key, _ in sorted_entries[:num_to_remove]:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            del self.cache_metadata[cache_key]
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
        
        logger.info(f"Evicted {num_to_remove} oldest cache entries")
        self._save_metadata()
    
    def _load_metadata(self) -> dict:
        """Load cache metadata from disk."""
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        metadata_file = self.cache_dir / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        # Clear memory
        self.memory_cache.clear()
        
        # Clear disk
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        self.cache_metadata.clear()
        self._save_metadata()
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_size = sum(m["size"] for m in self.cache_metadata.values())
        
        return {
            "entries": len(self.cache_metadata),
            "memory_entries": len(self.memory_cache),
            "total_size_mb": total_size / (1024 * 1024),
            "max_entries": self.max_size,
            "ttl_seconds": self.ttl,
            "usage_percentage": (len(self.cache_metadata) / self.max_size) * 100
        }


# Global cache instance
llm_cache = LLMCache()


def cached_llm_call(func: Callable) -> Callable:
    """Decorator to cache LLM calls."""
    
    @wraps(func)
    def wrapper(prompt: str, model: str = config.LLM_MODEL, **kwargs):
        # Try to get from cache
        cached_response = llm_cache.get(prompt, model, **kwargs)
        if cached_response is not None:
            return cached_response
        
        # Call function
        response = func(prompt, model, **kwargs)
        
        # Cache response
        llm_cache.set(prompt, model, response, **kwargs)
        
        return response
    
    return wrapper