#!/usr/bin/env python3
"""
Paper2Code - Production Entry Point
Converts research papers to executable code with comprehensive monitoring and error handling.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import config
from schema import GraphState
from logging_config import setup_logging, metrics_logger, error_tracker
from monitoring import metrics_collector, cost_tracker, performance_monitor
from rate_limiter import llm_rate_limiter, llm_circuit_breaker
from cache import llm_cache
from graph import app

logger = logging.getLogger(__name__)


class UserFeedback:
    """Collect and store user feedback."""
    
    def __init__(self):
        self.feedback_file = config.FEEDBACK_FILE
    
    def request_feedback(self, run_id: str, state: GraphState) -> dict:
        """Request feedback from user after run."""
        if not config.ENABLE_FEEDBACK_LOOP:
            return {}
        
        print("\n" + "=" * 80)
        print("FEEDBACK")
        print("=" * 80)
        print("Help us improve! Please provide feedback on this run.")
        print()
        
        feedback = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "success": state.get("error") is None,
        }
        
        try:
            # Quality rating
            rating = input("Rate quality (1-5, or press Enter to skip): ").strip()
            if rating:
                feedback["quality_rating"] = int(rating)
            
            # Accuracy rating
            accuracy = input("Rate accuracy vs paper (1-5, or press Enter to skip): ").strip()
            if accuracy:
                feedback["accuracy_rating"] = int(accuracy)
            
            # Comments
            comments = input("Comments (or press Enter to skip): ").strip()
            if comments:
                feedback["comments"] = comments
            
            # Would use again
            use_again = input("Would you use this again? (y/n, or press Enter to skip): ").strip().lower()
            if use_again:
                feedback["would_use_again"] = use_again == 'y'
            
            # Save feedback
            self._save_feedback(feedback)
            
            print("Thank you for your feedback!")
            
        except KeyboardInterrupt:
            print("\nFeedback cancelled.")
        except Exception as e:
            logger.warning(f"Failed to collect feedback: {e}")
        
        return feedback
    
    def _save_feedback(self, feedback: dict):
        """Save feedback to file."""
        try:
            with open(self.feedback_file, 'a') as f:
                f.write(json.dumps(feedback) + "\n")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")


class Paper2CodeRunner:
    """Main runner for Paper2Code pipeline."""
    
    def __init__(self):
        self.feedback_collector = UserFeedback()
        setup_logging()
    
    def run(self, paper_path: str, dry_run: bool = False) -> dict:
        """
        Run Paper2Code pipeline on a paper.
        
        Args:
            paper_path: Path to research paper
            dry_run: If True, validate but don't execute
            
        Returns:
            Final state dictionary
        """
        logger.info("=" * 80)
        logger.info("PAPER2CODE - PRODUCTION RUN")
        logger.info("=" * 80)
        logger.info(f"Paper: {paper_path}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Config: {config.LLM_MODEL} @ {config.LLM_TEMPERATURE}")
        logger.info("=" * 80)
        
        # Validate paper exists
        if not Path(paper_path).exists():
            logger.error(f"Paper not found: {paper_path}")
            return {"error": f"Paper not found: {paper_path}"}
        
        # Generate run ID
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start monitoring
        performance_monitor.checkpoint("start")
        cost_tracker.reset()
        metrics_collector.start_run(run_id, paper_path)
        
        # Print system status
        self._print_system_status()
        
        try:
            # Initialize state
            initial_state: GraphState = {
                "paper_path": paper_path,
                "recipe": None,
                "generated_code": None,
                "dockerfile_content": None,
                "docker_logs": None,
                "error": None,
                "mlflow_run_id": None,
                "retries": 0
            }
            
            if dry_run:
                logger.info("DRY RUN MODE - Validation only")
                return self._dry_run(initial_state)
            
            # Execute pipeline
            logger.info("Starting pipeline execution...")
            final_state = app.invoke(
                initial_state,
                {"recursion_limit": config.RECURSION_LIMIT}
            )
            
            # Determine success
            success = final_state.get("error") is None
            
            # End monitoring
            metrics_collector.end_run(
                "success" if success else "failed",
                error=Exception(final_state.get("error")) if not success else None
            )
            
            # Print results
            self._print_results(final_state, success)
            
            # Collect feedback
            if config.ENABLE_FEEDBACK_LOOP:
                self.feedback_collector.request_feedback(run_id, final_state)
            
            return final_state
        
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user")
            metrics_collector.end_run("interrupted")
            return {"error": "Interrupted by user"}
        
        except Exception as e:
            logger.critical(f"Pipeline crashed: {e}", exc_info=True)
            error_tracker.log_error(
                e,
                {"paper_path": paper_path, "run_id": run_id},
                "pipeline_crash",
                "critical"
            )
            metrics_collector.end_run("crashed", e)
            return {"error": f"Pipeline crashed: {e}"}
        
        finally:
            # Print final statistics
            self._print_statistics()
    
    def _dry_run(self, state: GraphState) -> dict:
        """Validate without executing."""
        logger.info("Validating paper...")
        
        from nodes import researcher_node
        result = researcher_node(state)
        
        if result.get("error"):
            logger.error(f"Validation failed: {result['error']}")
            return result
        
        logger.info("âœ… Validation passed")
        logger.info(f"Recipe: {json.dumps(result['recipe'], indent=2)}")
        
        return result
    
    def _print_system_status(self):
        """Print current system status."""
        print("\n" + "-" * 80)
        print("SYSTEM STATUS")
        print("-" * 80)
        
        # Rate limiter status
        rate_limit = llm_rate_limiter.get_current_usage()
        print(f"Rate Limit: {rate_limit['calls_used']}/{rate_limit['calls_available'] + rate_limit['calls_used']} calls available")
        
        # Circuit breaker status
        cb_state = llm_circuit_breaker.get_state()
        print(f"Circuit Breaker: {cb_state['state']} (failures: {cb_state['failure_count']})")
        
        # Cache status
        cache_stats = llm_cache.get_stats()
        print(f"Cache: {cache_stats['entries']} entries ({cache_stats['total_size_mb']:.2f} MB)")
        
        # Cost tracker
        budget_remaining = cost_tracker.get_remaining_budget()
        print(f"Budget: ${budget_remaining:.2f} remaining")
        
        print("-" * 80 + "\n")
    
    def _print_results(self, state: dict, success: bool):
        """Print pipeline results."""
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        if success:
            print("âœ… SUCCESS")
            print()
            print(f"MLflow Run ID: {state.get('mlflow_run_id')}")
            print(f"Generated code: {len(state.get('generated_code', ''))} characters")
            print(f"Retries: {state.get('retries', 0)}")
            
            if state.get("docker_logs"):
                print("\nDocker Logs (last 500 chars):")
                print("-" * 80)
                print(state["docker_logs"][-500:])
            
            print()
            print(f"View results: mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}")
        else:
            print("âŒ FAILED")
            print()
            print(f"Error: {state.get('error')}")
            print(f"Retries: {state.get('retries', 0)}")
            
            if state.get("docker_logs"):
                print("\nDocker Logs (last 1000 chars):")
                print("-" * 80)
                print(state["docker_logs"][-1000:])
        
        print("=" * 80 + "\n")
    
    def _print_statistics(self):
        """Print final statistics."""
        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)
        
        # Performance
        total_duration = performance_monitor.get_total_duration()
        print(f"Total Duration: {total_duration:.2f}s")
        
        # Metrics
        stats = metrics_collector.get_summary_stats()
        if "total_runs" in stats and stats["total_runs"] > 0:
            print(f"Success Rate: {stats['success_rate']:.1%}")
            print(f"Average Duration: {stats['avg_duration']:.2f}s")
            print(f"Average Retries: {stats['avg_retries']:.2f}")
        
        # Costs
        print(f"Cost This Run: ${cost_tracker.current_cost:.4f}")
        if "total_cost" in stats:
            print(f"Total Cost (All Runs): ${stats['total_cost']:.2f}")
        
        # Errors
        error_stats = error_tracker.get_error_stats()
        if error_stats:
            print("\nError Distribution:")
            for error_type, count in sorted(error_stats.items(), key=lambda x: -x[1])[:5]:
                print(f"  {error_type}: {count}")
        
        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Paper2Code: Convert research papers to executable code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.pdf                    # Process a paper
  %(prog)s paper.txt --dry-run          # Validate only
  %(prog)s paper.pdf --export-metrics   # Generate metrics report
  %(prog)s --test                       # Run test suite
  %(prog)s --clear-cache                # Clear LLM cache
        """
    )
    
    parser.add_argument(
        "paper_path",
        nargs="?",
        help="Path to research paper (PDF or TXT)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paper without executing pipeline"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test suite"
    )
    
    parser.add_argument(
        "--export-metrics",
        action="store_true",
        help="Export metrics report"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear LLM cache"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics from previous runs"
    )
    
    parser.add_argument(
        "--reset-circuit-breaker",
        action="store_true",
        help="Reset circuit breaker"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.test:
        from test_suite import run_tests
        setup_logging()
        success = run_tests()
        return 0 if success else 1
    
    if args.clear_cache:
        setup_logging()
        llm_cache.clear()
        print("âœ… Cache cleared")
        return 0
    
    if args.stats:
        setup_logging()
        stats = metrics_collector.get_summary_stats()
        print("\n" + json.dumps(stats, indent=2))
        return 0
    
    if args.export_metrics:
        setup_logging()
        report_path = metrics_collector.export_report()
        print(f"âœ… Metrics exported to: {report_path}")
        return 0
    
    if args.reset_circuit_breaker:
        setup_logging()
        llm_circuit_breaker.reset()
        print("âœ… Circuit breaker reset")
        return 0
    
    # Main pipeline execution
    if not args.paper_path:
        parser.print_help()
        return 1
    
    # Set verbose logging
    if args.verbose:
        config.LOG_LEVEL = logging.DEBUG
    
    # Run pipeline
    runner = Paper2CodeRunner()
    final_state = runner.run(args.paper_path, dry_run=args.dry_run)
    
    # Exit code
    success = final_state.get("error") is None
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)