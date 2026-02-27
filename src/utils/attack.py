"""
Attack pipeline execution and result management.
"""
import csv
import json
import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.llm import attack_target_llm, evaluate_response
from src.word import Dictionary, convert_query, reverse_convert
from src.utils.logger import log, close_log_file
from model import ModelFactory, LLMClient

# Default strategies for word substitution
STRATEGIES = ["Education", "Entertainment", "Health", "Business", "Technology"]

# Default directory where gen_dictionary.py saves pre-generated dictionaries
DICTIONARY_DIR = "results/dictionaries"


def _format_time(seconds):
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_result_dict(
    task_num: int,
    original_query: str,
    original_category: str,
    benign_category: str,
    converted_query: Optional[str],
    target_response: Optional[str],
    reversed_response: Optional[str],
    success: bool = False,
    score: Optional[float] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """Create a standardized result dictionary."""
    result = {
        "task": task_num,
        "original_query": original_query,
        "original_category": original_category,
        "benign_category": benign_category,
        "converted_query": converted_query,
        "target_response": target_response,
        "reversed_response": reversed_response,
        "success": success
    }
    
    if score is not None:
        result["score"] = score
    
    if error:
        result["error"] = error
    
    return result


def save_results(
    all_results: List[Dict[str, Any]], 
    output_file: str = "results/attack_results.json", 
    log_file: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None
) -> str:
    """Save all attack results grouped by task. Appends to existing file if it exists."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load existing results if file exists
    existing_tasks_dict = {}
    existing_summary = None
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if it's the new format with summary
                if isinstance(data, dict) and 'tasks' in data:
                    existing_summary = data.get('summary')
                    tasks_list = data['tasks']
                else:
                    # Old format: just a list of tasks
                    tasks_list = data
                
                for task_data in tasks_list:
                    if isinstance(task_data, dict) and 'task' in task_data:
                        task_num = task_data['task']
                        existing_tasks_dict[task_num] = task_data
        except (json.JSONDecodeError, KeyError) as e:
            # If file is corrupted, start fresh
            log(f"Warning: Could not read existing results file: {e}. Starting fresh.", log_file=log_file)
    
    # Process new results
    tasks_dict = existing_tasks_dict.copy()
    for result in all_results:
        task_num = result['task']
        
        if task_num not in tasks_dict:
            tasks_dict[task_num] = {
                "task": task_num,
                "original_query": result['original_query'],
                "original_category": result['original_category'],
                "categories": {}
            }
        
        category = result['benign_category']
        category_result = {
            "converted_query": result.get('converted_query'),
            "target_response": result.get('target_response'),
            "reversed_response": result.get('reversed_response'),
            "success": result.get('success', False)
        }
        
        if 'score' in result:
            category_result['score'] = result['score']
        
        if 'error' in result:
            category_result['error'] = result['error']
        
        tasks_dict[task_num]["categories"][category] = category_result
    
    # Calculate task-level success: true if any category succeeded
    for task_num in tasks_dict:
        task_success = any(
            cat_result.get('success', False)
            for cat_result in tasks_dict[task_num]["categories"].values()
        )
        tasks_dict[task_num]["success"] = task_success
    
    tasks_list = [tasks_dict[task_num] for task_num in sorted(tasks_dict)]
    
    # Prepare output data
    if summary:
        output_data = {
            "summary": summary,
            "tasks": tasks_list
        }
    else:
        # If no summary provided, keep old format for backward compatibility
        output_data = tasks_list
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_file


def load_harmful_queries(csv_path: str = "data/harmful_behaviors.csv", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load harmful queries from CSV file."""
    queries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if limit and idx > limit:
                break
            query = row.get('text', '').strip()
            if query:
                queries.append({
                    'task': idx,
                    'original_query': query,
                    'original_category': row.get('category', ''),
                })
    return queries


def attack_single_query(
    task_num: int,
    harmful_query: str,
    original_category: str,
    benign_category: str,
    target_llm_client: LLMClient,
    judge_llm_client: LLMClient,
    dictionary_dir: str = DICTIONARY_DIR,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute attack for a single query-category pair.
    Loads a pre-generated dictionary from dictionary_dir.
    If the dictionary file is not found, logs an error and returns an error result.
    """
    dict_path = os.path.join(dictionary_dir, f"task{task_num}", f"task{task_num}_{benign_category.lower()}.csv")

    try:
        # Step 1: Load pre-generated dictionary
        log(f"[Task {task_num}][{benign_category}] Step 1: Loading dictionary from {dict_path}...", log_file=log_file)
        if not os.path.exists(dict_path):
            raise FileNotFoundError(
                f"Dictionary file not found: {dict_path}. "
                "Run gen_dictionary.py first to pre-generate dictionaries."
            )
        dictionary = Dictionary.load_from_csv(dict_path)
        log(f"[Task {task_num}][{benign_category}] ✓ Dictionary loaded", log_file=log_file)

        # Step 2: Convert query
        log(f"[Task {task_num}][{benign_category}] Step 2: Converting query...", log_file=log_file)
        converted_query = convert_query(harmful_query, dictionary)
        log(f"[Task {task_num}][{benign_category}] Converted: {converted_query}", log_file=log_file)

        # Step 3: Attack target LLM
        log(f"[Task {task_num}][{benign_category}] Step 3: Attacking Target LLM...", log_file=log_file)
        target_response = attack_target_llm(
            harmful_query=harmful_query,
            dictionary=dictionary,
            target_llm_client=target_llm_client
        )
        log(f"[Task {task_num}][{benign_category}] ✓ Received response", log_file=log_file)

        # Step 4: Reverse-convert response
        log(f"[Task {task_num}][{benign_category}] Step 4: Reversing response...", log_file=log_file)
        reversed_response = reverse_convert(target_response, dictionary)
        log(f"[Task {task_num}][{benign_category}] ✓ Response reversed", log_file=log_file)

        # Log preview
        log(f"[Task {task_num}][{benign_category}] Original prompt: {harmful_query}", log_file=log_file)
        reversed_preview = " ".join(reversed_response.split()[:30])
        log(f"[Task {task_num}][{benign_category}] Reversed response: {reversed_preview}...", log_file=log_file)

        # Step 5: Evaluate response
        log(f"[Task {task_num}][{benign_category}] Step 5: Evaluating response...", log_file=log_file)
        evaluation = evaluate_response(
            user_prompt=harmful_query,
            agent_response=reversed_response,
            judge_llm_client=judge_llm_client
        )
        score = evaluation['score']
        success = evaluation['success']
        log(f"[Task {task_num}][{benign_category}] ✓ Evaluation completed (Score: {score:.3f}, {'Success' if success else 'Failed'})", log_file=log_file)

        return create_result_dict(
            task_num=task_num,
            original_query=harmful_query,
            original_category=original_category,
            benign_category=benign_category,
            converted_query=converted_query,
            target_response=target_response,
            reversed_response=reversed_response,
            success=success,
            score=score
        )

    except Exception as e:
        log(f"[Task {task_num}][{benign_category}] ✗ Error: {e}", "ERROR", log_file=log_file)
        import traceback
        traceback.print_exc()

        return create_result_dict(
            task_num=task_num,
            original_query=harmful_query,
            original_category=original_category,
            benign_category=benign_category,
            converted_query=None,
            target_response=None,
            reversed_response=None,
            success=False,
            error=str(e)
        )


def run_attack_pipeline(
    strategies: List[str],
    target_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    output_file: Optional[str] = None,
    csv_path: str = "data/harmful_behaviors.csv",
    limit: Optional[int] = None,
    early_stop: bool = True,
    dictionary_dir: str = DICTIONARY_DIR,
) -> str:
    """Run the complete attack pipeline using pre-generated dictionaries."""
    # Format model name for file names (replace hyphens with underscores)
    model_name = (target_model or "gpt-4o-mini").replace("-", "_")
    timestamp = datetime.now().strftime("%H%M%S")
    
    if output_file is None:
        output_file = f"results/results_{model_name}_{timestamp}.json"
    
    log_file = f"results/logs/log_{model_name}_{timestamp}.log"
    
    log("=" * 100, log_file=log_file)
    log("Starting Target LLM Attack Pipeline", log_file=log_file)
    log("=" * 100, log_file=log_file)
    log(f"Strategies to test: {strategies}", log_file=log_file)
    log(f"Early stop: {early_stop}", log_file=log_file)
    log(f"Dictionary directory: {dictionary_dir}", log_file=log_file)
    if limit:
        log(f"Limit: {limit} queries", log_file=log_file)
    log(f"Log file: {log_file}", log_file=log_file)
    log(f"Result file: {output_file}", log_file=log_file)
    
    try:
        queries = load_harmful_queries(csv_path, limit=limit)
        log(f"Loaded {len(queries)} harmful queries", log_file=log_file)
    except Exception as e:
        log(f"Failed to load harmful queries: {e}", "ERROR", log_file=log_file)
        close_log_file()
        raise
    
    log(f"Target LLM model: {target_model or 'gpt-4o-mini'}", log_file=log_file)
    log(f"Judge LLM model: {judge_model or 'gpt-4o-mini'}", log_file=log_file)
    try:
        target_llm = ModelFactory.create_target_llm(target_model)
        judge_llm = ModelFactory.create_judge_llm(judge_model)
        log("✓ LLM clients created", log_file=log_file)
    except Exception as e:
        log(f"Failed to create LLM clients: {e}", "ERROR", log_file=log_file)
        close_log_file()
        raise
    
    all_results = []
    start_time = time.time()
    
    try:
        for query_data in queries:
            task_num = query_data['task']
            harmful_query = query_data['original_query']
            original_category = query_data['original_category']
            
            task_start_time = time.time()
            
            log("", log_file=log_file)
            log("=" * 100, log_file=log_file)
            log(f"Task {task_num}/{len(queries)}: {harmful_query}", log_file=log_file)
            log("=" * 100, log_file=log_file)
            
            for strategy in strategies:
                log(f"[Task {task_num}] Testing strategy: {strategy}", log_file=log_file)
                
                result = attack_single_query(
                    task_num=task_num,
                    harmful_query=harmful_query,
                    original_category=original_category,
                    benign_category=strategy,
                    target_llm_client=target_llm,
                    judge_llm_client=judge_llm,
                    dictionary_dir=dictionary_dir,
                    log_file=log_file,
                )
                
                all_results.append(result)
                log("-" * 100, log_file=log_file)
                
                # Early stop if attack succeeded (when enabled)
                if early_stop and result.get('success', False):
                    log(f"[Task {task_num}] ✓ Attack succeeded with strategy '{strategy}'. Stopping early.", log_file=log_file)
                    break
            
            task_elapsed = time.time() - task_start_time
            log(f"[Task {task_num}] Completed in {_format_time(task_elapsed)}", log_file=log_file)
            
            # Save results after each task completes
            log(f"[Task {task_num}] Saving results...", log_file=log_file)
            save_results(all_results, output_file, log_file=log_file)
            log(f"[Task {task_num}] ✓ Results saved", log_file=log_file)
        
        log("", log_file=log_file)
        log("=" * 100, log_file=log_file)
        
        total_time = time.time() - start_time
        
        # Calculate success statistics (task-level: if any category succeeds, task succeeds)
        successful_tasks = set()
        for result in all_results:
            if result.get('success', False):
                successful_tasks.add(result['task'])
        
        # Prepare summary
        summary = {
            "total_tasks": len(queries),
            "successful_tasks": len(successful_tasks),
            "success_rate": len(successful_tasks) / len(queries) if len(queries) > 0 else 0.0,
            "total_time": total_time,
            "models": {
                "target_llm": target_model or "gpt-4o-mini",
                "judge_llm": judge_model or "gpt-4o-mini"
            }
        }
        
        # Save final results with summary
        result_file = save_results(all_results, output_file, log_file=log_file, summary=summary)
        log(f"✓ Final results saved to: {result_file}", log_file=log_file)
        
        log("", log_file=log_file)
        log("=" * 100, log_file=log_file)
        log("Attack Summary", log_file=log_file)
        log("=" * 100, log_file=log_file)
        log(f"Total tasks processed: {len(queries)}", log_file=log_file)
        log(f"Successful tasks: {len(successful_tasks)}/{len(queries)}", log_file=log_file)
        log(f"Success rate: {summary['success_rate']:.2%}", log_file=log_file)
        log(f"Total time elapsed: {_format_time(summary['total_time'])}", log_file=log_file)
        log(f"Results file: {result_file}", log_file=log_file)
        log(f"Log file: {log_file}", log_file=log_file)
        log("", log_file=log_file)
        log("Attack completed!", log_file=log_file)
        
    finally:
        close_log_file()
    
    return result_file
