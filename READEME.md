# Paper2Code - Production Edition

Convert research papers to executable code with comprehensive monitoring, error handling, and quality assurance.

## Features

✅ **Comprehensive Error Handling**
- Graceful degradation with intelligent fallbacks
- Detailed error tracking and categorization
- Automatic retry with exponential backoff

✅ **Advanced Monitoring**
- Real-time metrics collection
- Performance tracking per stage
- Cost tracking and budget management
- Success rate monitoring

✅ **Rate Limiting & Caching**
- Token bucket rate limiter
- Circuit breaker pattern
- LLM response caching (disk + memory)
- Cost optimization

✅ **Production-Ready Logging**
- Structured JSON logging
- Rotating log files
- Separate error logs
- Metrics logs for analysis

✅ **Comprehensive Testing**
- Unit tests for all components
- Integration tests
- Test suite with fixtures
- Automated testing support

✅ **User Feedback Loop**
- Post-run feedback collection
- Quality and accuracy ratings
- Comments and suggestions

## Installation
```bash
# Clone repository
git clone <repository-url>
cd paper2code

# Run setup script
chmod +x setup.sh
./setup.sh
```

## Quick Start
```bash
# Activate environment
source venv/bin/activate

# Process a paper
python -m main paper.pdf

# Dry run (validation only)
python -m main paper.txt --dry-run

# Run tests
python -m main --test

# View statistics
python -m main --stats
```

## Usage

### Basic Usage
```bash
python -m main path/to/paper.pdf
```

### Advanced Options
```bash
# Dry run - validate without executing
python -m main paper.pdf --dry-run

# Verbose logging
python -m main paper.pdf -v

# Clear cache
python -m main --clear-cache

# Reset circuit breaker
python -m main --reset-circuit-breaker

# Export metrics
python -m main --export-metrics
```

## Configuration

Edit `config.py` to customize:
```python
# LLM Settings
LLM_MODEL = "phi3"
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT = 300

# Rate Limiting
LLM_RATE_LIMIT_CALLS = 100
LLM_RATE_LIMIT_WINDOW = 3600

# Retry Settings
MAX_RETRIES = 3

# Timeouts (seconds)
TIMEOUTS = {
    "researcher": 300,
    "coder": 300,
    "qa": 1800,
}

# Caching
ENABLE_LLM_CACHE = True
CACHE_TTL = 86400  # 24 hours

# Cost Tracking
COST_BUDGET_PER_RUN = 1.0
```

## Architecture
```
Paper → Researcher → MLflow → Coder → QA → Success
           ↓          ↓        ↓       ↓
        Cache    Monitoring  Validate Docker
           ↓          ↓        ↓       ↓
      Rate Limit  Cost Track Retry  Error Track
```

## Monitoring
```bash
# View statistics
python -m main --stats

# Export detailed report
python -m main --export-metrics

# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Testing
```bash
# Run all tests
python -m main --test

# Run specific test
python -m test_suite TestPaper2Code.test_code_generation
```

## Troubleshooting

### Common Issues

**Issue**: Rate limit exceeded
```bash
python -m main --reset-circuit-breaker
```

**Issue**: Docker build fails
```bash
# Check Docker is running
docker ps

# Clear and rebuild
docker system prune -a
```

**Issue**: Out of memory
```bash
# Reduce batch size in config.py
# Or use smaller model
LLM_MODEL = "phi3:mini"
```

**Issue**: Slow performance
```bash
# Check cache hit rate
python -m main --stats

# Clear old cache if needed
python -m main --clear-cache
```

## File Structure
```
paper2code/
├── config.py              # Configuration
├── logging_config.py      # Logging system
├── monitoring.py          # Metrics
├── rate_limiter.py        # Rate limiting
├── cache.py               # Caching
├── schema.py              # Data schemas
├── prompts.py             # LLM prompts
├── templates.py           # Code templates
├── services.py            # Utilities
├── nodes.py               # Pipeline nodes
├── graph.py               # Workflow
├── main.py                # Entry point
├── test_suite.py          # Tests
├── requirements.txt       # Dependencies
├── setup.sh               # Setup script
└── README.md              # This file
```

## Support

- **Documentation**: This README
- **Issues**: GitHub Issues
- **Tests**: `python -m main --test`
- **Stats**: `python -m main --stats`

## License

MIT License

## Changelog

### v2.0.0 (Production Release)
- ✅ Comprehensive error handling
- ✅ Advanced monitoring and metrics
- ✅ Rate limiting and caching
- ✅ Full test coverage
- ✅ User feedback loop
- ✅ Production-ready logging
- ✅ Cost tracking
- ✅ Circuit breaker pattern