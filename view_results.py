import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("ðŸ“Š PAPER2CODE - TRAINING RESULTS")
print("="*80)

metrics_dir = Path("metrics")

if not metrics_dir.exists():
    print("âŒ No metrics directory found")
    exit()

run_files = sorted(list(metrics_dir.glob("run_*.json")), key=lambda x: x.stat().st_mtime, reverse=True)

if not run_files:
    print("âŒ No run files found")
    exit()

print(f"\nâœ… Found {len(run_files)} run(s)\n")

for idx, run_file in enumerate(run_files, 1):
    print("="*80)
    print(f"RUN #{idx}: {run_file.name}")
    print("="*80)
    
    with open(run_file, 'r') as f:
        data = json.load(f)
    
    # Basic Info
    print(f"\nðŸ“‹ BASIC INFO:")
    print(f"   Run ID: {data.get('run_id', 'N/A')}")
    print(f"   Status: {data.get('status', 'N/A').upper()}")
    print(f"   Paper: {data.get('paper_path', 'N/A')}")
    
    # Timing
    print(f"\nâ±ï¸  TIMING:")
    print(f"   Total Duration: {data.get('total_duration', 0):.2f}s")
    print(f"   Researcher: {data.get('researcher_duration', 0):.2f}s")
    print(f"   Coder: {data.get('coder_duration', 0):.2f}s")
    print(f"   QA: {data.get('qa_duration', 0):.2f}s")
    
    # Code Quality
    print(f"\nðŸ’» CODE GENERATION:")
    print(f"   Code Length: {data.get('code_length', 0)} characters")
    print(f"   Code Valid: {'âœ“' if data.get('code_valid') else 'âœ—'}")
    print(f"   Retries: {data.get('retries', 0)}")
    print(f"   Failed Stages: {', '.join(data.get('failed_stages', [])) if data.get('failed_stages') else 'None'}")
    
    # LLM Usage
    print(f"\nðŸ¤– LLM USAGE:")
    print(f"   API Calls: {data.get('llm_calls', 0)}")
    print(f"   Tokens: {data.get('llm_tokens', 0):,}")
    print(f"   Cost: ${data.get('llm_cost', 0):.4f}")
    
    # Status
    if data.get('status') == 'success':
        print(f"\nâœ… STATUS: SUCCESS")
    elif data.get('status') == 'failed':
        print(f"\nâŒ STATUS: FAILED")
        if data.get('error_message'):
            print(f"   Error Type: {data.get('error_type', 'Unknown')}")
            print(f"   Error: {data.get('error_message', 'No message')[:200]}")
    
    print()

print("="*80)
print("ðŸ“ˆ SUMMARY")
print("="*80)

# Calculate summary stats
total_runs = len(run_files)
successful_runs = sum(1 for f in run_files if json.load(open(f)).get('status') == 'success')
failed_runs = total_runs - successful_runs

if total_runs > 0:
    success_rate = (successful_runs / total_runs) * 100
    
    # Load all data for averages
    all_data = [json.load(open(f)) for f in run_files]
    avg_duration = sum(d.get('total_duration', 0) for d in all_data) / total_runs
    total_cost = sum(d.get('llm_cost', 0) for d in all_data)
    total_llm_calls = sum(d.get('llm_calls', 0) for d in all_data)
    
    print(f"\nTotal Runs: {total_runs}")
    print(f"Successful: {successful_runs} ({success_rate:.1f}%)")
    print(f"Failed: {failed_runs}")
    print(f"Average Duration: {avg_duration:.2f}s")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Total LLM Calls: {total_llm_calls}")

print("\n" + "="*80)

# Check for generated code
print("\nðŸ“ GENERATED CODE:")
build_dir = Path("build_context")
if (build_dir / "train.py").exists():
    with open(build_dir / "train.py", 'r') as f:
        code = f.read()
    print(f"   Location: {build_dir / 'train.py'}")
    print(f"   Size: {len(code)} characters")
    print(f"   Lines: {code.count(chr(10)) + 1}")
    
    # Show first few lines
    print(f"\n   First 10 lines:")
    for i, line in enumerate(code.split('\n')[:10], 1):
        print(f"      {i:2d}: {line[:70]}")
else:
    print("   No code found in build_context/")

# Check for recipe
print("\nðŸ“‹ RECIPE:")
if (build_dir / "recipe.json").exists():
    with open(build_dir / "recipe.json", 'r') as f:
        recipe = json.load(f)
    print(f"   Model: {recipe.get('model_architecture', 'N/A')}")
    print(f"   Dataset: {recipe.get('dataset', 'N/A')}")
    print(f"   Optimizer: {recipe.get('optimizer', 'N/A')}")
    print(f"   Learning Rate: {recipe.get('learning_rate', 'N/A')}")
    print(f"   Batch Size: {recipe.get('batch_size', 'N/A')}")
    print(f"   Loss Function: {recipe.get('loss_function', 'N/A')}")
else:
    print("   No recipe found")

print("\n" + "="*80)