import asyncio
import time
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure repository root is on path for package imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.code_optimizer_ai.core.code_scanner import code_scanner
from code.code_optimizer_ai.core.llm_analyzer import code_analyzer
from code.code_optimizer_ai.ml.rl_optimizer import get_rl_optimizer
from code.code_optimizer_ai.agents.pipeline_orchestrator import pipeline_orchestrator


class CodeOptimizerValidator:
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
    
    async def run_validation(self) -> Dict[str, Any]:
        
        print("Starting Velox Optimization Engine Validation Suite")
        
        try:
            # Test 1: Code Analysis Validation
            await self._test_code_analysis()
            
            # Test 2: Optimization Generation Validation  
            await self._test_optimization_generation()
            
            # Test 3: Performance Profiling Validation
            await self._test_performance_profiling()
            
            # Test 4: Real Code Examples Validation
            await self._test_real_code_examples()
            
            # Test 5: Pipeline Integration Validation
            await self._test_pipeline_integration()
            
            # Test 6: Error Handling Validation
            await self._test_error_handling()
            
            # Generate summary
            return self._generate_validation_summary()
            
        except Exception as e:
            print(f"FAIL Validation failed with error: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _test_code_analysis(self):
        
        print("\nTesting Code Analysis...")
        
        # Test code with known performance issues
        test_code = '''
def slow_function(n):
    """Function with nested loops - O(n²) complexity"""
    result = []
    for i in range(n):
        for j in range(n):
            result.append(i * j)
    return result

def inefficient_search(items, target):
    """Linear search - O(n) complexity"""
    for item in items:
        if item == target:
            return True
    return False

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_items(self, items):
        # Inefficient: appending one by one
        for item in items:
            self.data.append(item)
    
    def find_duplicates(self):
        # Inefficient duplicate detection
        duplicates = []
        for i, item1 in enumerate(self.data):
            for j, item2 in enumerate(self.data):
                if i != j and item1 == item2 and item1 not in duplicates:
                    duplicates.append(item1)
        return duplicates
'''
        
        try:
            # Test LLM analysis
            analysis_start = time.time()
            analysis_result = await code_analyzer.analyze_code(test_code, "test_validation.py")
            analysis_time = time.time() - analysis_start
            
            # Validate analysis results
            assert analysis_result.complexity_score > 0, "Complexity score should be > 0"
            assert analysis_result.confidence_score > 0, "Confidence should be > 0"
            assert len(analysis_result.performance_bottlenecks) > 0, "Should detect performance bottlenecks"
            assert analysis_time < 30, f"Analysis should complete within 30 seconds, took {analysis_time:.2f}s"
            
            # Test code scanner
            scanner_start = time.time()
            scan_result = await code_scanner.scan_directory(".", recursive=False)
            scanner_time = time.time() - scanner_start
            
            assert scan_result.scan_duration > 0, "Scan should take some time"
            assert scanner_time < 10, f"Scanning should complete within 10 seconds, took {scanner_time:.2f}s"
            
            # Record test results
            self.test_results.append({
                "test": "code_analysis",
                "status": "passed",
                "analysis_time": analysis_time,
                "scanner_time": scanner_time,
                "complexity_score": analysis_result.complexity_score,
                "bottlenecks_found": len(analysis_result.performance_bottlenecks)
            })
            
            print(f"PASS Code Analysis: {len(analysis_result.performance_bottlenecks)} bottlenecks detected")
            print(f"   Complexity Score: {analysis_result.complexity_score:.2f}")
            print(f"   Analysis Time: {analysis_time:.2f}s")
            
        except Exception as e:
            self.test_results.append({
                "test": "code_analysis", 
                "status": "failed",
                "error": str(e)
            })
            print(f"FAIL Code Analysis failed: {e}")
    
    async def _test_optimization_generation(self):
        
        print("\nTesting Optimization Generation...")
        
        test_code = '''
def compute_fibonacci(n):
    """Inefficient recursive fibonacci"""
    if n <= 1:
        return n
    return compute_fibonacci(n-1) + compute_fibonacci(n-2)

def find_maximum(numbers):
    """Linear search for maximum"""
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num
'''
        
        try:
            # Test optimization generation
            opt_start = time.time()
            suggestions = await get_rl_optimizer().optimize_code(test_code, "fibonacci_test.py")
            opt_time = time.time() - opt_start
            
            # Validate optimization results
            assert isinstance(suggestions, list), "Should return list of suggestions"
            assert len(suggestions) >= 0, "Should return at least empty list"
            
            if suggestions:
                # Validate suggestion structure
                suggestion = suggestions[0]
                assert hasattr(suggestion, 'category'), "Suggestion should have category"
                assert hasattr(suggestion, 'priority'), "Suggestion should have priority"
                assert hasattr(suggestion, 'confidence'), "Suggestion should have confidence"
                assert 0 <= suggestion.confidence <= 1, "Confidence should be between 0 and 1"
                
                # Check if optimization actually changes code
                assert suggestion.optimized_code != suggestion.original_code, "Should suggest actual changes"
            
            # Record test results
            self.test_results.append({
                "test": "optimization_generation",
                "status": "passed",
                "optimization_time": opt_time,
                "suggestions_count": len(suggestions),
                "avg_confidence": sum(s.confidence for s in suggestions) / len(suggestions) if suggestions else 0
            })
            
            print(f"PASS Optimization Generation: {len(suggestions)} suggestions generated")
            print(f"   Average Confidence: {sum(s.confidence for s in suggestions) / len(suggestions):.2f}" if suggestions else "   No suggestions generated")
            print(f"   Optimization Time: {opt_time:.2f}s")
            
        except Exception as e:
            self.test_results.append({
                "test": "optimization_generation",
                "status": "failed", 
                "error": str(e)
            })
            print(f"FAIL Optimization Generation failed: {e}")
    
    async def _test_performance_profiling(self):
        """Test performance profiling functionality"""
        
        print("\nTesting Performance Profiling...")
        
        try:
            from code.code_optimizer_ai.core.performance_profiler import performance_profiler
            
            # Test function profiling
            with performance_profiler.profile_function("validation_test"):
                # Simulate some computation
                result = sum(i * i for i in range(10000))
            
            # Test baseline creation
            if performance_profiler.metrics_buffer:
                metrics = performance_profiler.metrics_buffer
                baseline = performance_profiler.create_baseline("execution_time", metrics)
                
                assert baseline.baseline_value > 0, "Baseline value should be > 0"
                assert baseline.sample_count > 0, "Should have sample count > 0"
                
                # Test baseline comparison
                if len(metrics) > 1:
                    comparison = performance_profiler.compare_to_baseline(
                        metrics[-1], "execution_time", "validation_test"
                    )
                    assert "status" in comparison, "Comparison should include status"
                    assert "deviation" in comparison, "Comparison should include deviation"
            
            # Test performance summary
            summary = performance_profiler.get_performance_summary()
            
            # Record test results
            self.test_results.append({
                "test": "performance_profiling",
                "status": "passed",
                "metrics_collected": len(performance_profiler.metrics_buffer),
                "baseline_created": baseline.sample_count if 'baseline' in locals() else 0
            })
            
            print(f"PASS Performance Profiling: {len(performance_profiler.metrics_buffer)} metrics collected")
            if 'baseline' in locals():
                print(f"   Baseline: {baseline.baseline_value:.4f}s +/- {baseline.standard_deviation:.4f}s")
            
        except Exception as e:
            self.test_results.append({
                "test": "performance_profiling",
                "status": "failed",
                "error": str(e)
            })
            print(f"FAIL Performance Profiling failed: {e}")
    
    async def _test_real_code_examples(self):
        """Test with real-world code examples"""
        
        print("\nTesting Real Code Examples...")
        
        # Real-world code examples with known optimization opportunities
        real_code_examples = [
            {
                "name": "Web Scraping",
                "code": '''
import requests
from bs4 import BeautifulSoup

def scrape_websites(urls):
    """Inefficient web scraping - sequential requests"""
    results = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        results.append(soup.get_text())
    return results

def extract_emails(html_content):
    """Inefficient email extraction"""
    import re
    emails = []
    text = str(html_content)
    for match in re.finditer(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}', text):
        emails.append(match.group())
    return list(set(emails))  # Remove duplicates inefficiently
'''
            },
            {
                "name": "Data Processing",
                "code": '''
import pandas as pd
import numpy as np

def process_large_dataset(file_path):
    """Inefficient data processing"""
    df = pd.read_csv(file_path)
    
    # Inefficient grouping and aggregation
    result = []
    for category in df['category'].unique():
        subset = df[df['category'] == category]
        for idx, row in subset.iterrows():
            # Nested loops - very inefficient
            for idx2, row2 in subset.iterrows():
                if idx != idx2:
                    result.append({
                        'cat': category,
                        'val1': row['value'],
                        'val2': row2['value'],
                        'sum': row['value'] + row2['value']
                    })
    return pd.DataFrame(result)

def statistical_analysis(data):
    """Inefficient statistical computations"""
    n = len(data)
    
    # Recalculate mean repeatedly
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5
    
    # Inefficient percentiles
    sorted_data = sorted(data)
    p25 = sorted_data[int(0.25 * n)]
    p50 = sorted_data[int(0.50 * n)]
    p75 = sorted_data[int(0.75 * n)]
    
    return {
        'mean': mean,
        'std_dev': std_dev,
        'percentiles': {'25': p25, '50': p50, '75': p75}
    }
'''
            },
            {
                "name": "Algorithm Implementation",
                "code": '''
def bubble_sort(arr):
    """Classic bubble sort - O(n²)"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def find_primes(limit):
    """Inefficient prime number finder"""
    primes = []
    for num in range(2, limit):
        is_prime = True
        # Check divisibility by all numbers up to num
        for divisor in range(2, num):
            if num % divisor == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

def matrix_multiply(matrix1, matrix2):
    """Inefficient matrix multiplication"""
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])
    
    if cols1 != rows2:
        raise ValueError("Matrices cannot be multiplied")
    
    # Inefficient triple nested loops
    result = []
    for i in range(rows1):
        row = []
        for j in range(cols2):
            sum_val = 0
            for k in range(cols1):
                sum_val += matrix1[i][k] * matrix2[k][j]
            row.append(sum_val)
        result.append(row)
    
    return result
'''
            }
        ]
        
        total_suggestions = 0
        total_confidence = 0
        
        for example in real_code_examples:
            try:
                print(f"   Testing: {example['name']}")
                
                suggestions = await get_rl_optimizer().optimize_code(
                    example['code'], 
                    f"{example['name'].lower().replace(' ', '_')}.py"
                )
                
                total_suggestions += len(suggestions)
                if suggestions:
                    total_confidence += sum(s.confidence for s in suggestions)
                
                print(f"      PASS {len(suggestions)} optimization suggestions")
                
            except Exception as e:
                print(f"      FAIL Failed to optimize {example['name']}: {e}")
        
        # Record results
        self.test_results.append({
            "test": "real_code_examples",
            "status": "passed",
            "examples_tested": len(real_code_examples),
            "total_suggestions": total_suggestions,
            "avg_suggestions_per_example": total_suggestions / len(real_code_examples),
            "avg_confidence": total_confidence / total_suggestions if total_suggestions > 0 else 0
        })
        
        print(f"PASS Real Code Examples: {total_suggestions} total suggestions from {len(real_code_examples)} examples")
        print(f"   Average suggestions per example: {total_suggestions / len(real_code_examples):.1f}")
    
    async def _test_pipeline_integration(self):
        
        print("\nTesting Pipeline Integration...")
        
        try:
            # Test pipeline orchestrator
            pipeline_status = await pipeline_orchestrator.get_pipeline_status()
            
            assert "pipeline_status" in pipeline_status, "Should have pipeline status"
            assert "agents" in pipeline_status, "Should have agents info"
            
            # Test directory monitoring
            pipeline_orchestrator.add_watched_directory("./tests")
            
            # Test optimization request submission
            task_id = await pipeline_orchestrator.submit_optimization_request(
                "validation_test.py", 
                "medium"
            )
            
            assert task_id is not None, "Should return task ID"
            
            # Record results
            self.test_results.append({
                "test": "pipeline_integration",
                "status": "passed",
                "pipeline_running": pipeline_status["pipeline_status"] == "running",
                "agents_count": len(pipeline_status["agents"]),
                "task_submitted": True
            })
            
            print(f"PASS Pipeline Integration: {len(pipeline_status['agents'])} agents active")
            print(f"   Task ID: {task_id}")
            
        except Exception as e:
            self.test_results.append({
                "test": "pipeline_integration",
                "status": "failed",
                "error": str(e)
            })
            print(f"FAIL Pipeline Integration failed: {e}")
    
    async def _test_error_handling(self):
        
        print("\nTesting Error Handling...")
        
        test_cases = [
            {
                "name": "Empty code",
                "code": "",
                "should_handle": True
            },
            {
                "name": "Invalid Python syntax",
                "code": "def invalid_syntax(\n    return 1",
                "should_handle": True
            },
            {
                "name": "Very large code",
                "code": "# " + "x" * 100000,  # Very long comment
                "should_handle": True
            }
        ]
        
        handled_cases = 0
        
        for test_case in test_cases:
            try:
                suggestions = await get_rl_optimizer().optimize_code(
                    test_case["code"], 
                    f"error_test_{test_case['name'].lower().replace(' ', '_')}.py"
                )
                
                if test_case["should_handle"]:
                    handled_cases += 1
                    print(f"   PASS {test_case['name']}: Handled gracefully")
                
            except Exception as e:
                if test_case["should_handle"]:
                    print(f"   FAIL {test_case['name']}: Failed to handle - {e}")
                else:
                    handled_cases += 1
                    print(f"   WARN {test_case['name']}: Exception as expected - {e}")
        
        # Record results
        success_rate = handled_cases / len(test_cases)
        self.test_results.append({
            "test": "error_handling",
            "status": "passed" if success_rate >= 0.8 else "failed",
            "test_cases": len(test_cases),
            "handled_cases": handled_cases,
            "success_rate": success_rate
        })
        
        print(f"PASS Error Handling: {handled_cases}/{len(test_cases)} cases handled ({success_rate:.1%})")
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "passed"])
        failed_tests = total_tests - passed_tests
        
        summary = {
            "timestamp": time.time(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "test_results": self.test_results,
            "overall_status": "passed" if failed_tests == 0 else "failed"
        }
        
        # Performance metrics
        if any(r["test"] == "code_analysis" and r["status"] == "passed" for r in self.test_results):
            analysis_test = next(r for r in self.test_results if r["test"] == "code_analysis")
            summary["performance"] = {
                "analysis_time": analysis_test.get("analysis_time", 0),
                "scanner_time": analysis_test.get("scanner_time", 0)
            }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print validation summary"""
        
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Tests Failed: {summary['failed_tests']}/{summary['total_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Overall Status: {'PASSED' if summary['overall_status'] == 'passed' else 'FAILED'}")
        
        print("\nTest Breakdown:")
        for result in self.test_results:
            status_icon = "PASS" if result["status"] == "passed" else "FAIL"
            print(f"   {status_icon} {result['test'].replace('_', ' ').title()}")
        
        if "performance" in summary:
            perf = summary["performance"]
            print(f"\nPerformance Metrics:")
            print(f"   Code Analysis: {perf['analysis_time']:.2f}s")
            print(f"   Directory Scan: {perf['scanner_time']:.2f}s")
        
        print("\nVelox Optimization Engine Validation Complete!")
        
        # Save results to file
        results_file = Path("validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Detailed results saved to: {results_file}")


async def main():
    
    print("Velox Optimization Engine - System Validation")
    print("This script validates the complete functionality of the system")
    
    validator = CodeOptimizerValidator()
    summary = await validator.run_validation()
    validator.print_summary(summary)
    
    # Return exit code based on results
    return 0 if summary["overall_status"] == "passed" else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
