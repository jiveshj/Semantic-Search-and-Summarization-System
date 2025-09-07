# benchmark_system.py - Comprehensive benchmarking and testing suite
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import argparse
import logging
from datetime import datetime
import psutil
import threading
import concurrent.futures
from dataclasses import dataclass
import random
import requests
from pathlib import Path

from semantic_search_system import SemanticSearchSystem, Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    test_name: str
    num_documents: int
    num_queries: int
    avg_response_time: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    accuracy_score: float
    timestamp: datetime

class DatasetGenerator:
    """Generate synthetic datasets for benchmarking"""
    
    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Sample topics and content templates
        self.topics = [
            "machine learning", "artificial intelligence", "data science", 
            "computer vision", "natural language processing", "robotics",
            "cloud computing", "software engineering", "cybersecurity",
            "database systems", "web development", "mobile development"
        ]
        
        self.content_templates = [
            "{topic} is a rapidly evolving field that focuses on {description}. "
            "Key concepts include {concepts}. Applications range from {application1} "
            "to {application2}. Current challenges involve {challenge1} and {challenge2}.",
            
            "Recent advances in {topic} have revolutionized how we approach {problem}. "
            "Techniques such as {technique1} and {technique2} have shown promising results. "
            "Future research directions include {future1} and {future2}.",
            
            "The field of {topic} encompasses various methodologies including {method1}, "
            "{method2}, and {method3}. These approaches are particularly effective for "
            "{use_case}. Industry applications span {industry1} and {industry2}."
        ]
    
    def generate_document(self, doc_id: int) -> Document:
        """Generate a synthetic document"""
        topic = random.choice(self.topics)
        template = random.choice(self.content_templates)
        
        # Generate content based on template
        content = template.format(
            topic=topic,
            description=self._generate_description(topic),
            concepts=", ".join(self._generate_concepts(topic, 3)),
            application1=self._generate_application(topic),
            application2=self._generate_application(topic),
            challenge1=self._generate_challenge(topic),
            challenge2=self._generate_challenge(topic),
            problem=self._generate_problem(topic),
            technique1=self._generate_technique(topic),
            technique2=self._generate_technique(topic),
            future1=self._generate_future_direction(topic),
            future2=self._generate_future_direction(topic),
            method1=self._generate_method(topic),
            method2=self._generate_method(topic),
            method3=self._generate_method(topic),
            use_case=self._generate_use_case(topic),
            industry1=self._generate_industry(),
            industry2=self._generate_industry()
        )
        
        return Document(
            id=f"synthetic_{doc_id}",
            title=f"{topic.title()} Overview {doc_id}",
            content=content,
            source="synthetic",
            metadata={"topic": topic, "doc_id": doc_id}
        )
    
    def _generate_description(self, topic: str) -> str:
        descriptions = {
            "machine learning": "developing algorithms that learn from data",
            "artificial intelligence": "creating intelligent systems",
            "data science": "extracting insights from large datasets",
            "computer vision": "enabling machines to interpret visual information",
            "natural language processing": "teaching computers to understand human language",
            "robotics": "designing autonomous mechanical systems",
            "cloud computing": "delivering computing services over the internet",
            "software engineering": "systematic approaches to software development",
            "cybersecurity": "protecting digital systems from threats",
            "database systems": "efficiently storing and managing data",
            "web development": "creating applications for the internet",
            "mobile development": "building applications for mobile devices"
        }
        return descriptions.get(topic, "solving complex computational problems")
    
    def _generate_concepts(self, topic: str, num: int) -> List[str]:
        concept_map = {
            "machine learning": ["supervised learning", "unsupervised learning", "neural networks", "feature engineering"],
            "artificial intelligence": ["reasoning", "knowledge representation", "planning", "perception"],
            "data science": ["statistical analysis", "data mining", "visualization", "predictive modeling"],
            "computer vision": ["image recognition", "object detection", "feature extraction", "deep learning"],
            "natural language processing": ["tokenization", "sentiment analysis", "language models", "text classification"],
            "robotics": ["kinematics", "path planning", "sensor fusion", "control systems"],
            "cloud computing": ["virtualization", "scalability", "distributed systems", "microservices"],
            "software engineering": ["design patterns", "testing", "version control", "agile methodologies"],
            "cybersecurity": ["encryption", "threat detection", "access control", "vulnerability assessment"],
            "database systems": ["normalization", "indexing", "ACID properties", "query optimization"],
            "web development": ["HTML/CSS", "JavaScript frameworks", "REST APIs", "responsive design"],
            "mobile development": ["native apps", "cross-platform", "UI/UX design", "performance optimization"]
        }
        concepts = concept_map.get(topic, ["algorithms", "data structures", "optimization", "analysis"])
        return random.sample(concepts, min(num, len(concepts)))
    
    def _generate_application(self, topic: str) -> str:
        applications = {
            "machine learning": ["recommendation systems", "fraud detection", "image recognition"],
            "artificial intelligence": ["autonomous vehicles", "virtual assistants", "game playing"],
            "data science": ["market analysis", "customer segmentation", "risk assessment"],
            "computer vision": ["medical imaging", "security surveillance", "autonomous navigation"],
            "natural language processing": ["chatbots", "language translation", "document analysis"],
            "robotics": ["manufacturing automation", "healthcare assistance", "exploration"],
            "cloud computing": ["data storage", "application hosting", "disaster recovery"],
            "software engineering": ["enterprise applications", "mobile apps", "web services"],
            "cybersecurity": ["network protection", "data encryption", "threat monitoring"],
            "database systems": ["e-commerce", "financial systems", "content management"],
            "web development": ["social platforms", "e-commerce sites", "content portals"],
            "mobile development": ["gaming", "productivity apps", "social networking"]
        }
        apps = applications.get(topic, ["data processing", "system optimization"])
        return random.choice(apps)
    
    def _generate_challenge(self, topic: str) -> str:
        challenges = ["scalability", "accuracy", "performance", "security", "interpretability", 
                     "data quality", "real-time processing", "resource constraints"]
        return random.choice(challenges)
    
    def _generate_problem(self, topic: str) -> str:
        problems = ["complex optimization", "pattern recognition", "decision making", 
                   "resource allocation", "prediction accuracy", "system reliability"]
        return random.choice(problems)
    
    def _generate_technique(self, topic: str) -> str:
        techniques = ["deep learning", "ensemble methods", "optimization algorithms",
                     "statistical modeling", "reinforcement learning", "transfer learning"]
        return random.choice(techniques)
    
    def _generate_future_direction(self, topic: str) -> str:
        directions = ["explainable AI", "edge computing", "quantum computing integration",
                     "federated learning", "automated ML", "ethical AI development"]
        return random.choice(directions)
    
    def _generate_method(self, topic: str) -> str:
        methods = ["supervised learning", "unsupervised clustering", "reinforcement learning",
                  "transfer learning", "meta-learning", "self-supervised learning"]
        return random.choice(methods)
    
    def _generate_use_case(self, topic: str) -> str:
        use_cases = ["predictive analytics", "automated decision making", "pattern discovery",
                    "anomaly detection", "optimization problems", "classification tasks"]
        return random.choice(use_cases)
    
    def _generate_industry(self) -> str:
        industries = ["healthcare", "finance", "retail", "manufacturing", "transportation",
                     "entertainment", "education", "telecommunications", "energy"]
        return random.choice(industries)
    
    def generate_queries(self, num_queries: int) -> List[str]:
        """Generate test queries"""
        query_templates = [
            "What is {topic}?",
            "How does {topic} work?",
            "Applications of {topic} in {industry}",
            "Latest advances in {topic}",
            "Challenges in {topic}",
            "{topic} vs {other_topic}",
            "Best practices for {topic}",
            "Future of {topic}",
            "{topic} algorithms and techniques",
            "Real-world examples of {topic}"
        ]
        
        queries = []
        for _ in range(num_queries):
            template = random.choice(query_templates)
            topic = random.choice(self.topics)
            other_topic = random.choice([t for t in self.topics if t != topic])
            industry = self._generate_industry()
            
            query = template.format(topic=topic, other_topic=other_topic, industry=industry)
            queries.append(query)
        
        return queries

class SystemBenchmarker:
    """Comprehensive benchmarking suite for the semantic search system"""
    
    def __init__(self, search_system: SemanticSearchSystem):
        self.search_system = search_system
        self.dataset_generator = DatasetGenerator()
        self.results = []
    
    def benchmark_ingestion(self, num_documents: int, batch_size: int = 100) -> BenchmarkResult:
        """Benchmark document ingestion performance"""
        logger.info(f"Benchmarking ingestion of {num_documents} documents...")
        
        # Generate synthetic documents
        documents = [self.dataset_generator.generate_document(i) for i in range(num_documents)]
        
        # Monitor system resources
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the ingestion process
        start_time = time.time()
        
        # Add documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.content for doc in batch]
            embeddings = self.search_system.embedding_model.encode_documents(texts, batch_size)
            self.search_system.vector_index.add_documents(batch, embeddings)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        total_time = end_time - start_time
        throughput = num_documents / total_time
        memory_usage = end_memory - start_memory
        
        result = BenchmarkResult(
            test_name="document_ingestion",
            num_documents=num_documents,
            num_queries=0,
            avg_response_time=total_time / num_documents,
            throughput=throughput,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=psutil.cpu_percent(interval=1),
            accuracy_score=1.0,  # Perfect accuracy for ingestion
            timestamp=datetime.now()
        )
        
        self.results.append(result)
        logger.info(f"Ingestion completed: {throughput:.2f} docs/sec, {memory_usage:.2f}MB memory used")
        return result
    
    def benchmark_search_performance(self, num_queries: int, k: int = 10) -> BenchmarkResult:
        """Benchmark search query performance"""
        logger.info(f"Benchmarking {num_queries} search queries...")
        
        # Generate test queries
        queries = self.dataset_generator.generate_queries(num_queries)
        
        # Monitor system resources
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time search queries
        response_times = []
        start_time = time.time()
        
        for query in queries:
            query_start = time.time()
            results = self.search_system.search(query, k=k, include_summary=False)
            query_end = time.time()
            response_times.append(query_end - query_start)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_response_time = np.mean(response_times)
        throughput = num_queries / total_time
        memory_usage = end_memory - start_memory
        
        result = BenchmarkResult(
            test_name="search_performance",
            num_documents=len(self.search_system.vector_index.documents),
            num_queries=num_queries,
            avg_response_time=avg_response_time,
            throughput=throughput,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=psutil.cpu_percent(interval=1),
            accuracy_score=self._calculate_search_accuracy(queries[:100], k),
            timestamp=datetime.now()
        )
        
        self.results.append(result)
        logger.info(f"Search completed: {avg_response_time*1000:.2f}ms avg, {throughput:.2f} queries/sec")
        return result
    
    def benchmark_concurrent_load(self, num_threads: int, queries_per_thread: int) -> BenchmarkResult:
        """Benchmark concurrent query performance"""
        logger.info(f"Benchmarking concurrent load: {num_threads} threads, {queries_per_thread} queries each")
        
        # Generate test queries
        all_queries = self.dataset_generator.generate_queries(num_threads * queries_per_thread)
        
        # Monitor system resources
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        def worker_thread(queries: List[str]) -> List[float]:
            """Worker thread function"""
            times = []
            for query in queries:
                start_time = time.time()
                self.search_system.search(query, k=5, include_summary=False)
                end_time = time.time()
                times.append(end_time - start_time)
            return times
        
        # Execute concurrent queries
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Split queries among threads
            query_chunks = [all_queries[i::num_threads] for i in range(num_threads)]
            
            # Submit tasks
            futures = [executor.submit(worker_thread, chunk) for chunk in query_chunks]
            
            # Collect results
            all_response_times = []
            for future in concurrent.futures.as_completed(futures):
                all_response_times.extend(future.result())
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        total_time = end_time - start_time
        total_queries = num_threads * queries_per_thread
        avg_response_time = np.mean(all_response_times)
        throughput = total_queries / total_time
        memory_usage = end_memory - start_memory
        
        result = BenchmarkResult(
            test_name="concurrent_load",
            num_documents=len(self.search_system.vector_index.documents),
            num_queries=total_queries,
            avg_response_time=avg_response_time,
            throughput=throughput,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=psutil.cpu_percent(interval=1),
            accuracy_score=1.0,  # Not measuring accuracy for load test
            timestamp=datetime.now()
        )
        
        self.results.append(result)
        logger.info(f"Concurrent load completed: {throughput:.2f} queries/sec, {num_threads} threads")
        return result
    
    def _calculate_search_accuracy(self, queries: List[str], k: int) -> float:
        """Calculate search accuracy using synthetic relevance"""
        total_score = 0.0
        
        for query in queries[:20]:  # Sample for efficiency
            results = self.search_system.search(query, k=k, include_summary=False)
            
            # Simple relevance scoring based on keyword matching
            query_terms = set(query.lower().split())
            relevance_scores = []
            
            for result in results:
                doc_terms = set(result.document.content.lower().split())
                overlap = len(query_terms.intersection(doc_terms))
                relevance = overlap / len(query_terms) if query_terms else 0
                relevance_scores.append(relevance)
            
            # Calculate NDCG-like score
            if relevance_scores:
                dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
                ideal_relevance = sorted(relevance_scores, reverse=True)
                idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
                ndcg = dcg / idcg if idcg > 0 else 0
                total_score += ndcg
        
        return total_score / min(len(queries), 20)
    
    def benchmark_scalability(self, document_sizes: List[int], num_queries: int = 100) -> List[BenchmarkResult]:
        """Benchmark system scalability across different dataset sizes"""
        logger.info("Running scalability benchmark...")
        
        scalability_results = []
        
        for size in document_sizes:
            logger.info(f"Testing with {size} documents...")
            
            # Create fresh system for each test
            test_system = SemanticSearchSystem()
            test_benchmarker = SystemBenchmarker(test_system)
            
            # Ingest documents
            ingestion_result = test_benchmarker.benchmark_ingestion(size)
            
            # Test search performance
            search_result = test_benchmarker.benchmark_search_performance(num_queries)
            
            scalability_results.extend([ingestion_result, search_result])
        
        return scalability_results
    
    def generate_report(self, output_file: str = "benchmark_report.html"):
        """Generate comprehensive benchmark report"""
        if not self.results:
            logger.warning("No benchmark results to report")
            return
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            data.append({
                'Test Name': result.test_name,
                'Documents': result.num_documents,
                'Queries': result.num_queries,
                'Avg Response Time (ms)': result.avg_response_time * 1000,
                'Throughput (ops/sec)': result.throughput,
                'Memory Usage (MB)': result.memory_usage_mb,
                'CPU Usage (%)': result.cpu_usage_percent,
                'Accuracy Score': result.accuracy_score,
                'Timestamp': result.timestamp
            })
        
        df = pd.DataFrame(data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Response time vs dataset size
        if len(df[df['Test Name'] == 'search_performance']) > 1:
            search_data = df[df['Test Name'] == 'search_performance']
            axes[0, 0].plot(search_data['Documents'], search_data['Avg Response Time (ms)'], 'b-o')
            axes[0, 0].set_xlabel('Number of Documents')
            axes[0, 0].set_ylabel('Avg Response Time (ms)')
            axes[0, 0].set_title('Search Response Time vs Dataset Size')
            axes[0, 0].grid(True)
        
        # Throughput comparison
        throughput_data = df.groupby('Test Name')['Throughput (ops/sec)'].mean()
        axes[0, 1].bar(throughput_data.index, throughput_data.values)
        axes[0, 1].set_ylabel('Throughput (ops/sec)')
        axes[0, 1].set_title('Average Throughput by Test Type')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage
        axes[1, 0].plot(df['Documents'], df['Memory Usage (MB)'], 'g-o')
        axes[1, 0].set_xlabel('Number of Documents')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage vs Dataset Size')
        axes[1, 0].grid(True)
        
        # Accuracy scores
        accuracy_data = df[df['Accuracy Score'] > 0]['Accuracy Score']
        if len(accuracy_data) > 0:
            axes[1, 1].hist(accuracy_data, bins=10, alpha=0.7)
            axes[1, 1].set_xlabel('Accuracy Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Search Accuracy Distribution')
        
        plt.tight_layout()
        plt.savefig('benchmark_plots.png', dpi=300, bbox_inches='tight')
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Semantic Search System Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Semantic Search System Benchmark Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Total Tests:</strong> {len(self.results)}</p>
                <p><strong>Best Response Time:</strong> {df['Avg Response Time (ms)'].min():.2f} ms</p>
                <p><strong>Best Throughput:</strong> {df['Throughput (ops/sec)'].max():.2f} ops/sec</p>
                <p><strong>Average Accuracy:</strong> {df[df['Accuracy Score'] > 0]['Accuracy Score'].mean():.3f}</p>
            </div>
            
            <h2>Detailed Results</h2>
            {df.to_html(index=False, classes='table')}
            
            <h2>Performance Visualizations</h2>
            <img src="benchmark_plots.png" alt="Benchmark Plots" style="max-width: 100%;">
            
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Benchmark report generated: {output_file}")
        return df

class APILoadTester:
    """Load testing for the API service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_api_health(self) -> bool:
        """Test if API is healthy and responsive"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200 and response.json().get("status") == "healthy"
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
    
    def load_test_search(self, num_requests: int, concurrent_users: int) -> Dict:
        """Load test the search endpoint"""
        if not self.test_api_health():
            raise RuntimeError("API is not healthy")
        
        queries = [
            "machine learning algorithms",
            "artificial intelligence applications",
            "data science techniques",
            "computer vision methods",
            "natural language processing",
            "deep learning models",
            "cloud computing benefits",
            "software engineering practices"
        ]
        
        def make_request():
            query = random.choice(queries)
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/search",
                    json={"query": query, "k": 5, "include_summary": True},
                    timeout=30
                )
                end_time = time.time()
                return {
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "status_code": 0,
                    "response_time": 30.0,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute load test
        results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        end_time = time.time()
        
        # Calculate statistics
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
            stats = {
                "total_requests": num_requests,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / num_requests,
                "avg_response_time": np.mean(response_times),
                "min_response_time": np.min(response_times),
                "max_response_time": np.max(response_times),
                "p95_response_time": np.percentile(response_times, 95),
                "p99_response_time": np.percentile(response_times, 99),
                "requests_per_second": num_requests / (end_time - start_time),
                "total_time": end_time - start_time
            }
        else:
            stats = {
                "total_requests": num_requests,
                "successful_requests": 0,
                "failed_requests": len(failed_requests),
                "success_rate": 0.0,
                "error": "All requests failed"
            }
        
        return stats

def main():
    """Main benchmarking script"""
    parser = argparse.ArgumentParser(description="Benchmark the Semantic Search System")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of documents to test with")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of search queries to test")
    parser.add_argument("--concurrent_threads", type=int, default=10, help="Number of concurrent threads for load testing")
    parser.add_argument("--scalability_test", action="store_true", help="Run scalability tests with multiple dataset sizes")
    parser.add_argument("--api_load_test", action="store_true", help="Run API load tests")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if args.api_load_test:
        # API Load Testing
        logger.info("Starting API load tests...")
        load_tester = APILoadTester()
        
        if not load_tester.test_api_health():
            logger.error("API is not available for load testing")
            return
        
        # Run load test
        load_results = load_tester.load_test_search(
            num_requests=args.num_queries,
            concurrent_users=args.concurrent_threads
        )
        
        print("\n" + "="*50)
        print("API LOAD TEST RESULTS")
        print("="*50)
        for key, value in load_results.items():
            print(f"{key}: {value}")
        
        # Save results
        with open(f"{args.output_dir}/api_load_test.json", "w") as f:
            json.dump(load_results, f, indent=2, default=str)
    
    else:
        # System Benchmarking
        logger.info("Starting system benchmarks...")
        
        # Initialize system
        search_system = SemanticSearchSystem()
        benchmarker = SystemBenchmarker(search_system)
        
        if args.scalability_test:
            # Scalability testing
            sizes = [100, 500, 1000, 2000, 5000]
            results = benchmarker.benchmark_scalability(sizes, args.num_queries)
        else:
            # Standard benchmarking
            # 1. Document ingestion
            benchmarker.benchmark_ingestion(args.dataset_size)
            
            # 2. Search performance
            benchmarker.benchmark_search_performance(args.num_queries)
            
            # 3. Concurrent load
            benchmarker.benchmark_concurrent_load(args.concurrent_threads, args.num_queries // args.concurrent_threads)
        
        # Generate report
        report_file = f"{args.output_dir}/benchmark_report.html"
        df = benchmarker.generate_report(report_file)
        
        # Save raw results
        results_file = f"{args.output_dir}/benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump([
                {
                    "test_name": r.test_name,
                    "num_documents": r.num_documents,
                    "num_queries": r.num_queries,
                    "avg_response_time": r.avg_response_time,
                    "throughput": r.throughput,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                    "accuracy_score": r.accuracy_score,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in benchmarker.results
            ], f, indent=2)
        
        print("\n" + "="*50)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*50)
        print(df.groupby('Test Name').agg({
            'Avg Response Time (ms)': 'mean',
            'Throughput (ops/sec)': 'mean',
            'Memory Usage (MB)': 'mean',
            'Accuracy Score': 'mean'
        }).round(3))
        
        logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
