import os
import argparse
import time
from datetime import datetime
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
import asyncio
import requests

# Global logging setup for main function
global_log_file = None

def setup_global_logging(output_filename: str = None):
    """Setup global logging for main function."""
    global global_log_file
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract base name from output filename for log prefix
    if output_filename:
        # Remove path and extension to get clean base name
        base_name = os.path.splitext(os.path.basename(output_filename))[0]
        log_filename = os.path.join(logs_dir, f"{base_name}_main_{timestamp}.log")
    else:
        log_filename = os.path.join(logs_dir, f"paper_review_main_{timestamp}.log")
    
    global_log_file = open(log_filename, 'w', encoding='utf-8')
    return log_filename

def log_main(message: str):
    """Log message to both console and main log file."""
    print(message)
    if global_log_file:
        timestamp = datetime.now().strftime('%H:%M:%S')
        global_log_file.write(f"[{timestamp}] {message}\n")
        global_log_file.flush()

def close_global_logging():
    """Close the global log file."""
    global global_log_file
    if global_log_file:
        timestamp = datetime.now().strftime('%H:%M:%S')
        global_log_file.write(f"[{timestamp}] === Main Session Ended ===\n")
        global_log_file.close()
        global_log_file = None

def check_local_llm_server(url: str, model_name: str) -> bool:
    """Check if local LLM server is accessible and responding."""
    try:
        # Test the health endpoint or make a simple request
        test_url = f"{url}/v1/chat/completions"
        test_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }
        
        response = requests.post(
            test_url,
            json=test_payload,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return True
        else:
            print(f"❌ Local LLM server responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to local LLM server at {url}")
        print("   Make sure your LLM server is running and accessible")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Timeout connecting to local LLM server at {url}")
        return False
    except Exception as e:
        print(f"❌ Error testing local LLM server: {str(e)}")
        return False

def generate_output_filename(pdf_path: str, model_name: str) -> str:
    """Generate output filename from PDF name and model name."""
    import re
    
    # Extract filename without extension from PDF path
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Sanitize the filename - remove/replace invalid characters
    pdf_name = re.sub(r'[<>:"/\\|?*]', '', pdf_name)  # Remove invalid chars
    pdf_name = re.sub(r'\s+', '_', pdf_name)  # Replace spaces with underscores
    
    # Sanitize model name for filename (handle paths like meta-llama/Llama-3.1-8B-Instruct)
    safe_model_name = re.sub(r'[<>:"/\\|?*]', '-', model_name)  # Replace invalid chars with dash
    safe_model_name = re.sub(r'\s+', '_', safe_model_name)  # Replace spaces with underscores
    
    # Create the filename: pdf-name-model-name.md
    filename = f"{pdf_name}-{safe_model_name}.md"
    
    # Return full path in reviews folder
    return os.path.join("reviews", filename)

# Define the state structure for our agent
class PaperAnalysisState(TypedDict):
    pdf_path: str
    paper_content: str
    chunks: List[str]
    title: str
    authors: str
    summary: str
    contributions: str
    methodology: str
    results: str
    advantages: str
    disadvantages: str
    conclusion: str
    final_review: str
    current_step: str
    error: str
    # Performance tracking
    request_count: int
    step_times: Dict[str, float]
    step_requests: Dict[str, int]

class AcademicPaperAnalysisAgent:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.1, log_prefix: str = None, use_local_llm: bool = False, local_llm_url: str = "http://localhost:8080"):
        """
        Initialize the academic paper analysis agent.
        
        Args:
            model_name: The LLM model to use
            temperature: Temperature for LLM responses
            log_prefix: Optional prefix for log file naming (derived from output filename)
            use_local_llm: Whether to use a local LLM server
            local_llm_url: URL of the local LLM server
        """
        if use_local_llm:
            # Configure for local LLM server
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                base_url=f"{local_llm_url}/v1",  # Add /v1 to match OpenAI API format
                api_key="dummy-key"  # Local server doesn't need real API key
            )
        else:
            # Configure for OpenAI
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Reduced chunk size to handle large documents
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )
        self.log_prefix = log_prefix
        self.use_local_llm = use_local_llm
        self.workflow = self._build_workflow()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging to both console and dated log file."""
        # Create logs directory if it doesn't exist
        self.logs_dir = "logs"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        # Create dated log filename with optional prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.log_prefix:
            self.log_filename = os.path.join(self.logs_dir, f"{self.log_prefix}_{timestamp}.log")
        else:
            self.log_filename = os.path.join(self.logs_dir, f"paper_review_{timestamp}.log")
        
        self.log_file = open(self.log_filename, 'w', encoding='utf-8')
        
        # Log the start of the session
        self._log_to_file(f"=== Paper Review Session Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    def log(self, message: str):
        """Log message to both console and file."""
        print(message)
        self._log_to_file(message)
    
    def _log_to_file(self, message: str):
        """Write message to log file with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_file.write(f"[{timestamp}] {message}\n")
        self.log_file.flush()  # Ensure immediate write
    
    def close_log(self):
        """Close the log file."""
        if hasattr(self, 'log_file') and self.log_file:
            self._log_to_file("=== Paper Review Session Ended ===")
            self.log_file.close()
    
    def _track_step_performance(self, state: PaperAnalysisState, step_name: str, step_duration: float, step_requests: int, success: bool = True):
        """Helper method to track step performance metrics."""
        status = "completed" if success else "failed"
        self.log(f"⏱️  Step {status} in {step_duration:.2f}s ({step_requests} API requests)")
        
        # Track performance
        state["step_times"][step_name] = step_duration
        state["step_requests"][step_name] = step_requests
        state["request_count"] += step_requests
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for paper analysis."""
        workflow = StateGraph(PaperAnalysisState)
        
        # Add nodes for each step
        workflow.add_node("load_pdf", self.load_pdf)
        workflow.add_node("extract_title_authors", self.extract_title_authors)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("extract_contributions", self.extract_contributions)
        workflow.add_node("analyze_methodology", self.analyze_methodology)
        workflow.add_node("analyze_results", self.analyze_results)
        workflow.add_node("identify_advantages", self.identify_advantages)
        workflow.add_node("identify_disadvantages", self.identify_disadvantages)
        workflow.add_node("write_conclusion", self.write_conclusion)
        workflow.add_node("compile_review", self.compile_review)
        
        # Define the workflow edges
        workflow.set_entry_point("load_pdf")
        workflow.add_edge("load_pdf", "extract_title_authors")
        workflow.add_edge("extract_title_authors", "generate_summary")
        workflow.add_edge("generate_summary", "extract_contributions")
        workflow.add_edge("extract_contributions", "analyze_methodology")
        workflow.add_edge("analyze_methodology", "analyze_results")
        workflow.add_edge("analyze_results", "identify_advantages")
        workflow.add_edge("identify_advantages", "identify_disadvantages")
        workflow.add_edge("identify_disadvantages", "write_conclusion")
        workflow.add_edge("write_conclusion", "compile_review")
        workflow.add_edge("compile_review", END)
        
        return workflow.compile()
    
    async def load_pdf(self, state: PaperAnalysisState) -> PaperAnalysisState:
        """Step 1: Load and process the PDF content."""
        step_start = time.perf_counter()
        step_name = "load_pdf"
        try:
            self.log("📄 Loading PDF...")
            loader = PyPDFLoader(state["pdf_path"])
            documents = loader.load()
            
            # Combine all pages
            paper_content = "\n".join([doc.page_content for doc in documents])
            
            # Split into chunks for processing
            chunks = self.text_splitter.split_text(paper_content)
            
            # Log document info
            self.log(f"📊 Document loaded: {len(paper_content)} characters, {len(chunks)} chunks")
            if len(chunks) > 12:
                self.log(f"⚠️  Large document detected ({len(chunks)} chunks). Consider using --model gpt-3.5-turbo for better rate limits.")
            
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, 0)
            
            state.update({
                "paper_content": paper_content,
                "chunks": chunks,
                "current_step": "PDF loaded successfully"
            })
            
        except Exception as e:
            state["error"] = f"Error loading PDF: {str(e)}"
            
        return state
    
    async def extract_title_authors(self, state: PaperAnalysisState) -> PaperAnalysisState:
        """Step 1.5: Extract paper title and authors."""
        step_start = time.perf_counter()
        step_name = "extract_title_authors"
        step_requests = 0
        
        try:
            self.log("📌 Extracting title and authors...")
            
            prompt = ChatPromptTemplate.from_template("""
            You are an expert at extracting bibliographic information from academic papers. 
            Extract the title and authors from the following academic paper content.
            
            Focus on the first few pages which typically contain the title and author information.
            
            Paper content:
            {content}
            
            Please extract:
            1. Title: [The complete title of the paper]
            2. Authors: [List all authors, separated by commas]
            
            If you cannot find clear title or authors, indicate "Not clearly identifiable".
            
            Format your response as:
            Title: [title here]
            Authors: [authors here]
            """)
            
            # Use first 2-3 chunks which typically contain title/author info
            content_for_extraction = "\n".join(state["chunks"][:3])
            
            response = await self.llm.ainvoke(
                prompt.format_messages(content=content_for_extraction)
            )
            step_requests += 1
            
            # Parse the response to extract title and authors
            response_text = response.content
            title = "Not clearly identifiable"
            authors = "Not clearly identifiable"
            
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Title:'):
                    title = line[6:].strip()
                elif line.startswith('Authors:'):
                    authors = line[8:].strip()
            
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests)
            
            state.update({
                "title": title,
                "authors": authors,
                "current_step": "Title and authors extracted"
            })
            
        except Exception as e:
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests, success=False)
            
            state["error"] = f"Error extracting title and authors: {str(e)}"
            # Set defaults if extraction fails
            state.update({
                "title": "Title extraction failed",
                "authors": "Authors extraction failed"
            })
            
        return state
    
    async def generate_summary(self, state: PaperAnalysisState) -> PaperAnalysisState:
        """Step 2: Generate a 3-sentence easy-to-understand summary."""
        step_start = time.perf_counter()
        step_name = "generate_summary"
        step_requests = 0
        
        try:
            self.log("📝 Generating summary...")
            
            prompt = ChatPromptTemplate.from_template("""
            You are an expert academic paper reviewer. Read the following academic paper content and provide a clear, 3-sentence summary that would be understandable to someone not in the field.
            
            Focus on:
            - What problem the paper addresses
            - What approach they took
            - What they achieved/found
            
            Paper content:
            {content}
            
            Provide exactly 3 sentences that summarize this paper in simple, accessible language:
            """)
            
            # Use first few chunks for summary, but limit total length
            selected_chunks = state["chunks"][:4]
            content_for_summary = "\n".join(selected_chunks)
            
            # If still too long, use only first 2 chunks
            if len(content_for_summary) > 8000:
                content_for_summary = "\n".join(state["chunks"][:2])
            
            response = await self.llm.ainvoke(
                prompt.format_messages(content=content_for_summary)
            )
            step_requests += 1
            
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests)
            
            state.update({
                "summary": response.content,
                "current_step": "Summary generated"
            })
            
        except Exception as e:
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests, success=False)
            state["error"] = f"Error generating summary: {str(e)}"
            
        return state
    
    async def extract_contributions(self, state: PaperAnalysisState) -> PaperAnalysisState:
        """Step 3: Extract major contributions in bullet points."""
        step_start = time.perf_counter()
        step_name = "extract_contributions"
        step_requests = 0
        
        try:
            self.log("🎯 Extracting contributions...")
            
            prompt = ChatPromptTemplate.from_template("""
            Analyze the following academic paper and identify the major contributions. List them as clear bullet points.
            
            Look for:
            - Novel methods or algorithms
            - New theoretical insights
            - Empirical findings
            - Technical innovations
            - Improvements over existing work
            
            Paper content:
            {content}
            
            Major Contributions:
            """)
            
            # Use a representative sample of chunks to avoid token limits
            total_chunks = len(state["chunks"])
            if total_chunks <= 8:
                # Use all chunks if document is small
                content_for_analysis = "\n".join(state["chunks"])
            else:
                # For large documents, use strategic sampling:
                # - First 3 chunks (introduction/abstract)
                # - Middle 2 chunks (methodology/results)
                # - Last 2 chunks (conclusion/references)
                selected_indices = (
                    list(range(min(3, total_chunks))) +  # First chunks
                    [total_chunks // 2, total_chunks // 2 + 1] +  # Middle chunks
                    list(range(max(0, total_chunks - 2), total_chunks))  # Last chunks
                )
                # Remove duplicates and keep order
                selected_indices = sorted(list(set(selected_indices)))
                selected_chunks = [state["chunks"][i] for i in selected_indices if i < total_chunks]
                content_for_analysis = "\n".join(selected_chunks)
            
            response = await self.llm.ainvoke(
                prompt.format_messages(content=content_for_analysis)
            )
            step_requests += 1
            
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests)
            
            state.update({
                "contributions": response.content,
                "current_step": "Contributions extracted"
            })
            
        except Exception as e:
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests, success=False)
            state["error"] = f"Error extracting contributions: {str(e)}"
            
        return state
    
    async def analyze_methodology(self, state: PaperAnalysisState) -> PaperAnalysisState:
        """Step 4: Summarize main methodology."""
        step_start = time.perf_counter()
        step_name = "analyze_methodology"
        step_requests = 0
        
        try:
            self.log("🔬 Analyzing methodology...")
            
            prompt = ChatPromptTemplate.from_template("""
            Based on the major contributions identified below, analyze and summarize the main methodology used in this paper.
            
            Major Contributions:
            {contributions}
            
            Paper content:
            {content}
            
            Focus on:
            - Key methodological approaches
            - Technical frameworks used
            - Experimental design (if applicable)
            - How the methodology supports the contributions
            
            Main Methodology:
            """)
            
            # Use a representative sample of chunks to avoid token limits
            total_chunks = len(state["chunks"])
            if total_chunks <= 8:
                content_for_analysis = "\n".join(state["chunks"])
            else:
                # Focus on methodology sections (typically in middle of paper)
                mid_point = total_chunks // 2
                start_idx = max(0, mid_point - 3)
                end_idx = min(total_chunks, mid_point + 3)
                selected_chunks = state["chunks"][start_idx:end_idx]
                # Also include first chunk for context
                if start_idx > 0:
                    selected_chunks = [state["chunks"][0]] + selected_chunks
                content_for_analysis = "\n".join(selected_chunks)
            
            response = await self.llm.ainvoke(
                prompt.format_messages(
                    contributions=state["contributions"],
                    content=content_for_analysis
                )
            )
            step_requests += 1
            
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests)
            
            state.update({
                "methodology": response.content,
                "current_step": "Methodology analyzed"
            })
            
        except Exception as e:
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests, success=False)
            state["error"] = f"Error analyzing methodology: {str(e)}"
            
        return state
    
    async def analyze_results(self, state: PaperAnalysisState) -> PaperAnalysisState:
        """Step 5: Summarize main data results supporting contributions."""
        step_start = time.perf_counter()
        step_name = "analyze_results"
        step_requests = 0
        
        try:
            self.log("📊 Analyzing results...")
            
            prompt = ChatPromptTemplate.from_template("""
            Analyze the results section and identify the main data results that support each major contribution.
            
            Major Contributions:
            {contributions}
            
            Paper content:
            {content}
            
            For each contribution, identify supporting data/results where available:
            - Quantitative results (accuracy, performance metrics, etc.)
            - Qualitative findings
            - Comparative results
            - Statistical significance
            
            Main Results Supporting Contributions:
            """)
            
            # Use a representative sample of chunks to avoid token limits
            total_chunks = len(state["chunks"])
            if total_chunks <= 8:
                content_for_analysis = "\n".join(state["chunks"])
            else:
                # Focus on results sections (typically in latter half of paper)
                mid_point = total_chunks // 2
                start_idx = mid_point
                end_idx = min(total_chunks, mid_point + 4)
                selected_chunks = state["chunks"][start_idx:end_idx]
                # Also include abstract/intro for context
                if len(state["chunks"]) > 0:
                    selected_chunks = [state["chunks"][0]] + selected_chunks
                content_for_analysis = "\n".join(selected_chunks)
            
            response = await self.llm.ainvoke(
                prompt.format_messages(
                    contributions=state["contributions"],
                    content=content_for_analysis
                )
            )
            step_requests += 1
            
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests)
            
            state.update({
                "results": response.content,
                "current_step": "Results analyzed"
            })
            
        except Exception as e:
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests, success=False)
            state["error"] = f"Error analyzing results: {str(e)}"
            
        return state
    
    async def identify_advantages(self, state: PaperAnalysisState) -> PaperAnalysisState:
        """Step 6: Identify 3 advantages/pros of the paper."""
        step_start = time.perf_counter()
        step_name = "identify_advantages"
        step_requests = 0
        
        try:
            self.log("✅ Identifying advantages...")
            
            prompt = ChatPromptTemplate.from_template("""
            Based on your analysis of this academic paper, identify exactly 3 key advantages or strengths of this work.
            
            Paper Summary: {summary}
            Contributions: {contributions}
            Methodology: {methodology}
            Results: {results}
            
            Consider:
            - Novelty and innovation
            - Methodological rigor
            - Practical applicability
            - Theoretical contributions
            - Empirical validation
            - Clarity of presentation
            
            List exactly 3 advantages:
            """)
            
            response = await self.llm.ainvoke(
                prompt.format_messages(
                    summary=state["summary"],
                    contributions=state["contributions"],
                    methodology=state["methodology"],
                    results=state["results"]
                )
            )
            step_requests += 1
            
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests)
            
            state.update({
                "advantages": response.content,
                "current_step": "Advantages identified"
            })
            
        except Exception as e:
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests, success=False)
            state["error"] = f"Error identifying advantages: {str(e)}"
            
        return state
    
    async def identify_disadvantages(self, state: PaperAnalysisState) -> PaperAnalysisState:
        """Step 7: Identify 3 disadvantages/cons of the paper."""
        step_start = time.perf_counter()
        step_name = "identify_disadvantages"
        step_requests = 0
        
        try:
            self.log("❌ Identifying disadvantages...")
            
            prompt = ChatPromptTemplate.from_template("""
            Based on your analysis of this academic paper, identify exactly 3 key disadvantages, limitations, or areas for improvement.
            
            Paper Summary: {summary}
            Contributions: {contributions}
            Methodology: {methodology}
            Results: {results}
            
            Consider:
            - Methodological limitations
            - Scope constraints
            - Experimental limitations
            - Generalizability issues
            - Missing comparisons
            - Unclear explanations
            - Future work needed
            
            List exactly 3 disadvantages or limitations:
            """)
            
            response = await self.llm.ainvoke(
                prompt.format_messages(
                    summary=state["summary"],
                    contributions=state["contributions"],
                    methodology=state["methodology"],
                    results=state["results"]
                )
            )
            step_requests += 1
            
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests)
            
            state.update({
                "disadvantages": response.content,
                "current_step": "Disadvantages identified"
            })
            
        except Exception as e:
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests, success=False)
            state["error"] = f"Error identifying disadvantages: {str(e)}"
            
        return state
    
    async def write_conclusion(self, state: PaperAnalysisState) -> PaperAnalysisState:
        """Step 8: Write 1-2 sentence conclusion."""
        step_start = time.perf_counter()
        step_name = "write_conclusion"
        step_requests = 0
        
        try:
            self.log("🎯 Writing conclusion...")
            
            prompt = ChatPromptTemplate.from_template("""
            Based on the complete analysis of this academic paper, write a concise 1-2 sentence conclusion that captures the overall assessment of the work.
            
            Summary: {summary}
            Contributions: {contributions}
            Advantages: {advantages}
            Disadvantages: {disadvantages}
            
            Write a balanced, professional conclusion (1-2 sentences):
            """)
            
            response = await self.llm.ainvoke(
                prompt.format_messages(
                    summary=state["summary"],
                    contributions=state["contributions"],
                    advantages=state["advantages"],
                    disadvantages=state["disadvantages"]
                )
            )
            step_requests += 1
            
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests)
            
            state.update({
                "conclusion": response.content,
                "current_step": "Conclusion written"
            })
            
        except Exception as e:
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests, success=False)
            state["error"] = f"Error writing conclusion: {str(e)}"
            
        return state
    
    async def compile_review(self, state: PaperAnalysisState) -> PaperAnalysisState:
        """Step 9: Compile all analysis into a single page review."""
        step_start = time.perf_counter()
        step_name = "compile_review"
        step_requests = 0
        
        try:
            self.log("📋 Compiling final review...")
            
            review_template = """
# Academic Paper Review

**Title:** {title}  
**Authors:** {authors}

## Summary
{summary}

## Major Contributions
{contributions}

## Methodology
{methodology}

## Results and Data
{results}

## Advantages
{advantages}

## Disadvantages
{disadvantages}

## Conclusion
{conclusion}

---
*Review generated by Academic Paper Analysis Agent*
            """
            
            final_review = review_template.format(
                title=state["title"],
                authors=state["authors"],
                summary=state["summary"],
                contributions=state["contributions"],
                methodology=state["methodology"],
                results=state["results"],
                advantages=state["advantages"],
                disadvantages=state["disadvantages"],
                conclusion=state["conclusion"]
            )
            
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests)
            
            state.update({
                "final_review": final_review,
                "current_step": "Review compiled successfully"
            })
            
        except Exception as e:
            step_duration = time.perf_counter() - step_start
            self._track_step_performance(state, step_name, step_duration, step_requests, success=False)
            state["error"] = f"Error compiling review: {str(e)}"
            
        return state
    
    async def analyze_paper(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main method to analyze an academic paper.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing all analysis results
        """
        # Initialize state
        initial_state = PaperAnalysisState(
            pdf_path=pdf_path,
            paper_content="",
            chunks=[],
            title="",
            authors="",
            summary="",
            contributions="",
            methodology="",
            results="",
            advantages="",
            disadvantages="",
            conclusion="",
            final_review="",
            current_step="Starting analysis...",
            error="",
            # Performance tracking
            request_count=0,
            step_times={},
            step_requests={}
        )
        
        # Run the workflow
        self.log("🚀 Starting academic paper analysis workflow...")
        final_state = await self.workflow.ainvoke(initial_state)
        
        if final_state.get("error"):
            self.log(f"❌ Error occurred: {final_state['error']}")
            return {"error": final_state["error"]}
        
        self.log("✅ Analysis completed successfully!")
        
        # Print performance summary
        total_time = sum(final_state["step_times"].values())
        total_requests = final_state["request_count"]
        self.log(f"\n📊 Performance Summary:")
        self.log(f"   Total time: {total_time:.2f}s")
        self.log(f"   Total API requests: {total_requests}")
        self.log(f"   Average time per request: {total_time/max(total_requests, 1):.2f}s")
        
        self.log("\n🔍 Step breakdown:")
        for step_name, duration in final_state["step_times"].items():
            requests = final_state["step_requests"].get(step_name, 0)
            self.log(f"   {step_name}: {duration:.2f}s ({requests} requests)")
        
        return {
            "title": final_state["title"],
            "authors": final_state["authors"],
            "summary": final_state["summary"],
            "contributions": final_state["contributions"],
            "methodology": final_state["methodology"],
            "results": final_state["results"],
            "advantages": final_state["advantages"],
            "disadvantages": final_state["disadvantages"],
            "conclusion": final_state["conclusion"],
            "final_review": final_state["final_review"]
        }
    
    def save_review(self, review_content: str, output_path: str):
        """Save the review to a markdown file."""
        # Create reviews directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.log(f"📁 Created directory: {output_dir}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(review_content)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Academic Paper Analysis Agent - Analyze academic papers using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper-review.py papers/my_paper.pdf
    → Output: reviews/my_paper-gpt-4.md
  
  python paper-review.py papers/research.pdf --model gpt-3.5-turbo
    → Output: reviews/research-gpt-3.5-turbo.md
  
  python paper-review.py papers/study.pdf --output reviews/custom_name.md
    → Output: reviews/custom_name.md (custom filename)
  
  python paper-review.py papers/analysis.pdf --temperature 0.2
    → Output: reviews/analysis-gpt-4.md
    
  python paper-review.py papers/analysis.pdf --local-llm --model meta-llama/Llama-3.1-8B-Instruct
    → Output: reviews/analysis-meta-llama-Llama-3.1-8B-Instruct.md (using local LLM)
        """
    )
    
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to analyze"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Model to use (default: gpt-4). For OpenAI: gpt-4, gpt-3.5-turbo, gpt-4-turbo. For local LLM: meta-llama/Llama-3.1-8B-Instruct or any model name your server supports"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for AI responses (0.0-1.0, default: 0.1)"
    )
    
    parser.add_argument(
        "--output",
        default=None,
        help="Output file name (default: auto-generated from PDF and model name)"
    )
    
    parser.add_argument(
        "--local-llm",
        action="store_true",
        help="Use local LLM server instead of OpenAI"
    )
    
    parser.add_argument(
        "--local-url",
        default="http://localhost:8080",
        help="URL of local LLM server (default: http://localhost:8080)"
    )
    
    return parser.parse_args()

async def main():
    """Main function to run the Academic Paper Analysis Agent."""
    
    main_log_file = None
    
    try:
        # Parse command line arguments first
        args = parse_arguments()
        
        # Validate PDF file exists first (before setting up logging)
        if not os.path.exists(args.pdf_path):
            print(f"❌ Error: PDF file not found: {args.pdf_path}")
            print("Please check the file path and try again.")
            return
        
        if not args.pdf_path.lower().endswith('.pdf'):
            print(f"❌ Error: File must be a PDF: {args.pdf_path}")
            return
        
        # Determine output filename early for log naming
        if args.output is None:
            output_path = generate_output_filename(args.pdf_path, args.model)
        else:
            output_path = args.output
        
        # Extract base name for log prefix
        log_prefix = os.path.splitext(os.path.basename(output_path))[0]
        
        # Setup global logging with prefix
        main_log_file = setup_global_logging(output_path)
        log_main(f"=== Paper Review Main Session Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        
        # Load API key from .env file if it exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            log_main("💡 Tip: Install python-dotenv to use .env files: pip install python-dotenv")
        
        # Check if API key is set (only for OpenAI, not needed for local LLM)
        if not args.local_llm and not os.environ.get("OPENAI_API_KEY"):
            log_main("❌ Error: OPENAI_API_KEY not found!")
            log_main("Please set your OpenAI API key:")
            log_main("  1. Set environment variable: export OPENAI_API_KEY='your-key-here'")
            log_main("  2. Create .env file with: OPENAI_API_KEY=your-key-here")
            log_main("  3. Uncomment and edit the line above with your API key")
            log_main("  4. Or use --local-llm to use your local LLM server")
            return
        
        # Initialize the agent with command line arguments and log prefix
        if args.local_llm:
            log_main(f"🤖 Initializing agent with local model: {args.model} at {args.local_url}")
            
            # Test local LLM server connectivity
            log_main("🔌 Testing connection to local LLM server...")
            if not check_local_llm_server(args.local_url, args.model):
                log_main("❌ Failed to connect to local LLM server. Please check:")
                log_main(f"   1. Server is running at {args.local_url}")
                log_main(f"   2. Model '{args.model}' is loaded")
                log_main("   3. Server is accessible from this machine")
                return
            log_main("✅ Local LLM server connection successful")
        else:
            log_main(f"🤖 Initializing agent with OpenAI model: {args.model}")
            
        agent = AcademicPaperAnalysisAgent(
            model_name=args.model,
            temperature=args.temperature,
            log_prefix=log_prefix,
            use_local_llm=args.local_llm,
            local_llm_url=args.local_url
        )
        
        log_main(f"📄 Analyzing PDF: {args.pdf_path}")
        if args.output is None:
            log_main(f"📝 Auto-generated output filename: {output_path}")
        log_main(f"📝 Agent log file: {agent.log_filename}")
        log_main(f"📝 Main log file: {main_log_file}")
        
        try:
            # Analyze the paper
            results = await agent.analyze_paper(args.pdf_path)
            
            if "error" in results:
                log_main(f"❌ Analysis failed: {results['error']}")
                return
            
            # Save to file
            agent.save_review(results["final_review"], output_path)
            log_main(f"💾 Review saved to: {output_path}")
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                log_main(f"❌ Rate limit error: {error_msg}")
                log_main("💡 Suggestions:")
                log_main("   1. Use: python paper-review.py papers/prefill-only.pdf --model gpt-3.5-turbo")
                log_main("   2. Wait a few minutes and try again")
                log_main("   3. Check your OpenAI usage limits at https://platform.openai.com/account/usage")
            else:
                log_main(f"❌ Error: {str(e)}")
                log_main("Please check your OpenAI API key and try again.")
        finally:
            # Close agent log file if agent was created
            if 'agent' in locals():
                agent.close_log()
            
    finally:
        # Close global log file if it was set up
        if main_log_file:
            close_global_logging()

# Usage:
"""
python paper-review.py <pdf_path> [options]

Examples:
  # Using OpenAI models (requires API key)
  python paper-review.py papers/research_paper.pdf
    → Output: reviews/research_paper-gpt-4.md
  
  python paper-review.py papers/study.pdf --model gpt-3.5-turbo
    → Output: reviews/study-gpt-3.5-turbo.md
  
  python paper-review.py papers/analysis.pdf --output reviews/custom_name.md
    → Output: reviews/custom_name.md (custom filename)
  
  # Using local LLM server (no API key needed)
  python paper-review.py papers/research_paper.pdf --local-llm --model meta-llama/Llama-3.1-8B-Instruct
    → Output: reviews/research_paper-meta-llama-Llama-3.1-8B-Instruct.md
  
  python paper-review.py papers/study.pdf --local-llm --local-url http://localhost:8080
    → Uses local LLM server at specified URL

Requirements:
  pip install langgraph langchain langchain-openai langchain-community pypdf python-dotenv

For OpenAI models, you'll need to set your API key:
  export OPENAI_API_KEY="your-api-key-here"
  OR create a .env file with: OPENAI_API_KEY=your-api-key-here

For local LLM, make sure your server is running and accessible at the specified URL.
"""

if __name__ == "__main__":
    # Set your OpenAI API key - uncomment and add your key, or set as environment variable
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Run the analysis
    asyncio.run(main())