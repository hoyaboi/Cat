"""
Attack pipeline execution and result management.
"""
import csv
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.llm import generate_dictionary, attack_target_llm
from src.word import convert_query, reverse_convert
from src.utils.logger import log, close_log_file
from model import ModelFactory, LLMClient


def create_result_dict(
    task_num: int,
    original_query: str,
    original_category: str,
    benign_category: str,
    converted_query: Optional[str],
    target_response: Optional[str],
    reversed_response: Optional[str],
    success: bool = False,
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
    
    if error:
        result["error"] = error
    
    return result


def save_results(all_results: List[Dict[str, Any]], output_file: str = "results/attack_results.json") -> str:
    """Save all attack results grouped by task."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    tasks_dict = {}
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
        
        if 'error' in result:
            category_result['error'] = result['error']
        
        tasks_dict[task_num]["categories"][category] = category_result
    
    tasks_list = [tasks_dict[task_num] for task_num in sorted(tasks_dict)]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tasks_list, f, indent=2, ensure_ascii=False)
    
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
    word_llm_client: LLMClient,
    target_llm_client: LLMClient,
    dictionary_dir: str = "results/dictionaries",
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """Execute attack for a single query-category pair."""
    try:
        log(f"[Task {task_num}][{benign_category}] Step 1: Generating dictionary...", log_file=log_file)
        dictionary = generate_dictionary(
            harmful_query=harmful_query,
            target_category=benign_category,
            word_llm_client=word_llm_client,
            output_dir=dictionary_dir,
            task_num=task_num
        )
        log(f"[Task {task_num}][{benign_category}] ✓ Dictionary generated", log_file=log_file)
        
        log(f"[Task {task_num}][{benign_category}] Step 2: Converting query...", log_file=log_file)
        converted_query = convert_query(harmful_query, dictionary)
        log(f"[Task {task_num}][{benign_category}] Converted: {converted_query}", log_file=log_file)
        
        log(f"[Task {task_num}][{benign_category}] Step 3: Attacking Target LLM...", log_file=log_file)
        target_response = attack_target_llm(
            harmful_query=harmful_query,
            dictionary=dictionary,
            target_llm_client=target_llm_client
        )
        log(f"[Task {task_num}][{benign_category}] ✓ Received response", log_file=log_file)
        
        log(f"[Task {task_num}][{benign_category}] Step 4: Reversing response...", log_file=log_file)
        reversed_response = reverse_convert(target_response, dictionary)
        log(f"[Task {task_num}][{benign_category}] ✓ Response reversed", log_file=log_file)
        
        return create_result_dict(
            task_num=task_num,
            original_query=harmful_query,
            original_category=original_category,
            benign_category=benign_category,
            converted_query=converted_query,
            target_response=target_response,
            reversed_response=reversed_response,
            success=False
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
    categories: List[str],
    model_name: str,
    output_file: Optional[str] = None,
    csv_path: str = "data/harmful_behaviors.csv",
    limit: Optional[int] = None
) -> str:
    """Run the complete attack pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_file is None:
        output_file = f"results/result_{timestamp}.json"
    
    log_file = f"results/log_{timestamp}.log"
    
    log("=" * 80, log_file=log_file)
    log("Starting Target LLM Attack Pipeline", log_file=log_file)
    log("=" * 80, log_file=log_file)
    log(f"Categories to test: {categories}", log_file=log_file)
    if limit:
        log(f"Limit: {limit} queries", log_file=log_file)
    log(f"Log file: {log_file}", log_file=log_file)
    log(f"Result file: {output_file}", log_file=log_file)
    
    dictionary_base_dir = f"results/dictionaries/dictionary_{timestamp}"
    os.makedirs(dictionary_base_dir, exist_ok=True)
    log(f"Dictionary directory: {dictionary_base_dir}", log_file=log_file)
    
    try:
        queries = load_harmful_queries(csv_path, limit=limit)
        log(f"Loaded {len(queries)} harmful queries", log_file=log_file)
    except Exception as e:
        log(f"Failed to load harmful queries: {e}", "ERROR", log_file=log_file)
        close_log_file()
        raise
    
    log(f"Using model: {model_name} for Word LLM and Target LLM", log_file=log_file)
    try:
        word_llm = ModelFactory.create_word_llm(model_name)
        target_llm = ModelFactory.create_target_llm(model_name)
        log(f"✓ LLM clients created", log_file=log_file)
    except Exception as e:
        log(f"Failed to create LLM clients: {e}", "ERROR", log_file=log_file)
        close_log_file()
        raise
    
    all_results = []
    
    try:
        for query_data in queries:
            task_num = query_data['task']
            harmful_query = query_data['original_query']
            original_category = query_data['original_category']
            
            log("", log_file=log_file)
            log("-" * 80, log_file=log_file)
            log(f"Task {task_num}/{len(queries)}: {harmful_query}", log_file=log_file)
            log("-" * 80, log_file=log_file)
            
            task_dict_dir = os.path.join(dictionary_base_dir, f"task{task_num}")
            os.makedirs(task_dict_dir, exist_ok=True)
            
            for benign_category in categories:
                log(f"[Task {task_num}] Testing category: {benign_category}", log_file=log_file)
                
                result = attack_single_query(
                    task_num=task_num,
                    harmful_query=harmful_query,
                    original_category=original_category,
                    benign_category=benign_category,
                    word_llm_client=word_llm,
                    target_llm_client=target_llm,
                    dictionary_dir=task_dict_dir,
                    log_file=log_file
                )
                
                all_results.append(result)
            
            if task_num % 10 == 0 or task_num == len(queries):
                log(f"[Task {task_num}] Saving intermediate results...", log_file=log_file)
                save_results(all_results, output_file)
                log(f"[Task {task_num}] ✓ Results saved", log_file=log_file)
        
        log("", log_file=log_file)
        log("=" * 80, log_file=log_file)
        log("Saving final results...", log_file=log_file)
        result_file = save_results(all_results, output_file)
        log(f"✓ Final results saved to: {result_file}", log_file=log_file)
        
        log("", log_file=log_file)
        log("=" * 80, log_file=log_file)
        log("Attack Summary", log_file=log_file)
        log("=" * 80, log_file=log_file)
        log(f"Total tasks processed: {len(queries)}", log_file=log_file)
        log(f"Total results: {len(all_results)}", log_file=log_file)
        log(f"Expected results: {len(queries) * len(categories)}", log_file=log_file)
        log(f"Results file: {result_file}", log_file=log_file)
        log(f"Log file: {log_file}", log_file=log_file)
        log("", log_file=log_file)
        log("Attack completed!", log_file=log_file)
        
    finally:
        close_log_file()
    
    return result_file
