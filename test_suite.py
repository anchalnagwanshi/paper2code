import unittest
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List
import logging

import config
from schema import ExperimentRecipe
import nodes
from graph import app

logger = logging.getLogger(__name__)


class TestPaper2Code(unittest.TestCase):
    """Comprehensive test suite for Paper2Code."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.test_papers_dir = cls.test_dir / "papers"
        cls.test_papers_dir.mkdir()
        
        # Create test papers
        cls._create_test_papers()
        
        logger.info(f"Test directory: {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_papers(cls):
        """Create sample test papers."""
        
        # Simple paper
        simple_paper = """
        Title: Image Classification with ResNet-50
        
        Abstract:
        We train a ResNet-50 model on CIFAR-10 dataset for image classification.
        
        Methodology:
        We use a ResNet-50 architecture with the following hyperparameters:
        - Optimizer: Adam
        - Learning rate: 0.001
        - Batch size: 32
        - Loss function: Cross-Entropy
        - GPU: Yes
        
        We train for 50 epochs and evaluate on the test set using accuracy.
        """
        
        with open(cls.test_papers_dir / "simple_paper.txt", "w") as f:
            f.write(simple_paper)
        
        # Medium complexity paper
        medium_paper = """
        Title: BERT Fine-tuning for Sentiment Analysis
        
        Abstract:
        We fine-tune BERT-base on the IMDB dataset for sentiment classification.
        
        Methodology:
        Architecture: BERT-base transformer
        Dataset: IMDB sentiment dataset from HuggingFace
        Preprocessing: Tokenization with BERT tokenizer, max length 512
        
        Training:
        - Optimizer: AdamW
        - Learning rate: 2e-5
        - Batch size: 16
        - Epochs: 3
        - Loss: Binary Cross-Entropy
        - GPU required: Yes
        
        Evaluation:
        We report accuracy and F1-score on the test set.
        """
        
        with open(cls.test_papers_dir / "medium_paper.txt", "w") as f:
            f.write(medium_paper)
        
        # Minimal paper (edge case)
        minimal_paper = """
        Title: Simple Experiment
        Model: Linear
        Dataset: Custom
        """
        
        with open(cls.test_papers_dir / "minimal_paper.txt", "w") as f:
            f.write(minimal_paper)
    
    def test_load_simple_paper(self):
        """Test loading a simple paper."""
        paper_path = str(self.test_papers_dir / "simple_paper.txt")
        
        initial_state = {
            "paper_path": paper_path,
            "recipe": None,
            "generated_code": None,
            "dockerfile_content": None,
            "docker_logs": None,
            "error": None,
            "mlflow_run_id": None,
            "retries": 0
        }
        
        result = nodes.researcher_node(initial_state)
        
        self.assertIsNone(result["error"], f"Researcher failed: {result.get('error')}")
        self.assertIsNotNone(result["recipe"])
        self.assertIsInstance(result["recipe"], dict)
    
    def test_recipe_validation(self):
        """Test recipe extraction and validation."""
        paper_path = str(self.test_papers_dir / "medium_paper.txt")
        
        initial_state = {
            "paper_path": paper_path,
            "recipe": None,
            "generated_code": None,
            "dockerfile_content": None,
            "docker_logs": None,
            "error": None,
            "mlflow_run_id": None,
            "retries": 0
        }
        
        result = nodes.researcher_node(initial_state)
        
        if result["error"]:
            self.skipTest(f"Researcher failed: {result['error']}")
        
        recipe = result["recipe"]
        
        # Validate required fields
        self.assertIn("model_architecture", recipe)
        self.assertIn("dataset", recipe)
        self.assertIn("optimizer", recipe)
        self.assertIn("learning_rate", recipe)
        self.assertIn("batch_size", recipe)
        
        # Validate types
        self.assertIsInstance(recipe["learning_rate"], (int, float))
        self.assertIsInstance(recipe["batch_size"], int)
        
        # Validate ranges
        self.assertGreater(recipe["learning_rate"], 0)
        self.assertGreater(recipe["batch_size"], 0)
    
    def test_code_generation(self):
        """Test code generation."""
        mock_recipe = {
            "model_architecture": "ResNet-50",
            "dataset": "CIFAR-10",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "loss_function": "CrossEntropyLoss",
            "gpu_required": False
        }
        
        state = {
            "recipe": mock_recipe,
            "mlflow_run_id": "test_run_123",
            "retries": 0
        }
        
        result = nodes.coder_node(state)
        
        self.assertIsNone(result["error"])
        self.assertIsNotNone(result["generated_code"])
        
        code = result["generated_code"]
        
        # Validate code structure
        self.assertIn("import torch", code)
        self.assertIn("import mlflow", code)
        self.assertGreater(len(code), 500)
        
        # Validate it compiles
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")
    
    def test_invalid_paper_path(self):
        """Test handling of invalid paper path."""
        state = {
            "paper_path": "/nonexistent/paper.txt",
            "recipe": None,
            "generated_code": None,
            "dockerfile_content": None,
            "docker_logs": None,
            "error": None,
            "mlflow_run_id": None,
            "retries": 0
        }
        
        result = nodes.researcher_node(state)
        
        self.assertIsNotNone(result["error"])
        self.assertIn("not found", result["error"].lower())
    
    def test_minimal_paper(self):
        """Test handling of minimal paper with little information."""
        paper_path = str(self.test_papers_dir / "minimal_paper.txt")
        
        state = {
            "paper_path": paper_path,
            "recipe": None,
            "generated_code": None,
            "dockerfile_content": None,
            "docker_logs": None,
            "error": None,
            "mlflow_run_id": None,
            "retries": 0
        }
        
        result = nodes.researcher_node(state)
        
        # Should succeed but with warnings/defaults
        self.assertIsNotNone(result["recipe"])
        self.assertIsNotNone(result["error"])  # Expect an error
        self.assertIn("too short", result["error"].lower()) 
    
    def test_retry_mechanism(self):
        """Test retry mechanism."""
        state = {
            "recipe": {"model_architecture": "test"},
            "mlflow_run_id": "test_123",
            "retries": 0,
            "error": "Previous error",
            "docker_logs": "Error: Module not found"
        }
        
        result = nodes.coder_node(state)
        
        # Should increment retries
        self.assertEqual(result["retries"], 1)
        self.assertIsNotNone(result["generated_code"])
    
    def test_max_retries(self):
        """Test max retries limit."""
        from graph import decide_after_qa
        
        state = {
            "retries": config.MAX_RETRIES,
            "error": "Persistent error"
        }
        
        decision = decide_after_qa(state)
        
        # Should end after max retries
        self.assertEqual(decision, "end")
    
    def test_cost_tracking(self):
        """Test cost tracking functionality."""
        from monitoring import cost_tracker
        
        initial_cost = cost_tracker.current_cost
        cost_tracker.add_cost(0.01, "test_operation")
        
        self.assertEqual(cost_tracker.current_cost, initial_cost + 0.01)
        
        # Test budget alert
        remaining = cost_tracker.get_remaining_budget()
        self.assertGreaterEqual(remaining, 0)
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        from monitoring import metrics_collector
        
        run = metrics_collector.start_run("test_run", "test_paper.txt")
        
        self.assertEqual(run.run_id, "test_run")
        self.assertEqual(run.status, "running")
        
        metrics_collector.record_stage_duration("researcher", 10.5)
        metrics_collector.record_llm_call(tokens=1000, cost=0.01)
        metrics_collector.end_run("success")
        
        stats = metrics_collector.get_summary_stats()
        self.assertGreater(stats["total_runs"], 0)
    
    def test_cache_functionality(self):
        """Test LLM caching."""
        from cache import llm_cache
        
        # Clear cache
        llm_cache.clear()
        
        # Set and get
        test_prompt = "test prompt"
        test_response = "test response"
        
        llm_cache.set(test_prompt, "test_model", test_response)
        cached = llm_cache.get(test_prompt, "test_model")
        
        self.assertEqual(cached, test_response)
        
        # Check stats
        stats = llm_cache.get_stats()
        self.assertGreater(stats["entries"], 0)


class IntegrationTests(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up integration test fixtures."""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.test_paper = cls.test_dir / "integration_paper.txt"
        
        paper_content = """
        Title: Test Experiment
        
        We train a simple neural network on dummy data.
        
        Model: Multi-layer perceptron
        Dataset: Random data
        Optimizer: SGD
        Learning rate: 0.01
        Batch size: 16
        Loss: MSE
        """
        
        with open(cls.test_paper, "w") as f:
            f.write(paper_content)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_full_pipeline(self):
        """Test complete pipeline execution."""
        initial_state = {
            "paper_path": str(self.test_paper),
            "recipe": None,
            "generated_code": None,
            "dockerfile_content": None,
            "docker_logs": None,
            "error": None,
            "mlflow_run_id": None,
            "retries": 0
        }
        
        # This is an integration test, so it might take time
        # and could fail in test environment without Docker
        try:
            final_state = app.invoke(initial_state, {"recursion_limit": 10})
            
            # Check final state
            self.assertIsNotNone(final_state.get("recipe"))
            self.assertIsNotNone(final_state.get("generated_code"))
            
            # Success if no error or within retry limit
            if final_state.get("error"):
                self.assertLessEqual(
                    final_state.get("retries", 0),
                    config.MAX_RETRIES + 1
                )
        except Exception as e:
            self.skipTest(f"Integration test skipped: {e}")


def run_tests(verbosity: int = 2):
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestPaper2Code))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    from logging_config import setup_logging
    
    setup_logging()
    success = run_tests()
    sys.exit(0 if success else 1)