# Paper Review Agent

An intelligent academic paper analysis agent built with LangGraph that automatically reviews academic papers and generates comprehensive reviews.

## Features

- 📄 **PDF Processing**: Automatically loads and processes academic papers in PDF format
- 📝 **Comprehensive Analysis**: Generates summaries, extracts contributions, analyzes methodology and results
- ✅ **Pros & Cons**: Identifies advantages and disadvantages of the research
- 📋 **Structured Review**: Compiles all analysis into a well-formatted markdown review
- 🔄 **LangGraph Workflow**: Uses a structured workflow for systematic paper analysis
- 🏠 **Local LLM Support**: Use your own local LLM server (e.g., Llama models) instead of OpenAI
- 💰 **Cost Flexible**: Choose between OpenAI API (pay-per-use) or local models (free after setup)

## Setup

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd paper-review-agent
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv paper-review-env

# Activate it (Linux/Mac)
source paper-review-env/bin/activate

# Or use the provided script
source activate.sh
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up LLM Backend

#### Option A: OpenAI API (Recommended for beginners)
You need an OpenAI API key to use OpenAI models.

Create a `.env` file in the project root:
```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

Or export it directly:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

#### Option B: Local LLM Server (Free, no API key needed)
Set up a local LLM server that supports OpenAI-compatible API. Popular options:

- **vLLM**: Fast inference server for Llama models
- **Ollama**: Easy-to-use local LLM runner
- **text-generation-webui**: Web interface with API support

Example with vLLM serving Llama-3.1-8B-Instruct:
```bash
# Install vLLM
pip install vllm

# Start server (example)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8080
```

Make sure your local server is accessible at `http://localhost:8080` (or your preferred URL).

### Which Option Should You Choose?

| Feature | OpenAI API | Local LLM |
|---------|------------|-----------|
| **Setup Difficulty** | Easy (just API key) | Moderate (server setup) |
| **Cost** | Pay-per-use | Free after setup |
| **Performance** | Excellent | Good (depends on model/hardware) |
| **Privacy** | Data sent to OpenAI | Data stays local |
| **Internet Required** | Yes | No |
| **Resource Usage** | None | High (GPU/RAM) |

## Usage

### Basic Command Line Usage

#### Using OpenAI Models
```bash
# Analyze a paper with default settings (GPT-4, temperature 0.1)
python paper-review.py papers/your_paper.pdf

# Use a faster/cheaper model
python paper-review.py papers/your_paper.pdf --model gpt-3.5-turbo

# Customize output file and temperature
python paper-review.py papers/your_paper.pdf --output reviews/custom_review.md --temperature 0.2
```

#### Using Local LLM (No API key needed)
```bash
# Use local Llama model (default URL: http://localhost:8080)
python paper-review.py papers/your_paper.pdf --local-llm --model meta-llama/Llama-3.1-8B-Instruct

# Specify custom local server URL
python paper-review.py papers/your_paper.pdf --local-llm --local-url http://localhost:8080 --model meta-llama/Llama-3.1-8B-Instruct

# With custom output filename
python paper-review.py papers/your_paper.pdf --local-llm --model meta-llama/Llama-3.1-8B-Instruct --output reviews/llama_review.md
```

### Command Line Options
```bash
python paper-review.py <pdf_path> [options]

Required:
  pdf_path              Path to the PDF file to analyze

Optional:
  --model MODEL         Model to use (default: gpt-4)
                        For OpenAI: gpt-4, gpt-3.5-turbo, gpt-4-turbo
                        For local LLM: meta-llama/Llama-3.1-8B-Instruct (or any model your server supports)
  --temperature TEMP    Temperature 0.0-1.0 for AI responses (default: 0.1)
  --output FILE         Output filename (default: auto-generated from PDF and model name)
  --local-llm           Use local LLM server instead of OpenAI
  --local-url URL       URL of local LLM server (default: http://localhost:8080)
  --help               Show help message
```

### Programmatic Usage

#### Using OpenAI Models
```python
import asyncio
from paper_review_agent import AcademicPaperAnalysisAgent

async def analyze_paper():
    agent = AcademicPaperAnalysisAgent(model_name="gpt-4", temperature=0.1)
    results = await agent.analyze_paper("path/to/your/paper.pdf")
    agent.save_review(results["final_review"], "reviews/my_review.md")

asyncio.run(analyze_paper())
```

#### Using Local LLM
```python
import asyncio
from paper_review_agent import AcademicPaperAnalysisAgent

async def analyze_paper():
    agent = AcademicPaperAnalysisAgent(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.1,
        use_local_llm=True,
        local_llm_url="http://localhost:8080"
    )
    results = await agent.analyze_paper("path/to/your/paper.pdf")
    agent.save_review(results["final_review"], "reviews/llama_review.md")

asyncio.run(analyze_paper())
```

## Analysis Components

The agent performs the following analysis steps:

1. **Title & Authors**: Extracts paper title and author list
2. **Summary**: 3-sentence easy-to-understand summary
3. **Contributions**: Major contributions in bullet points
4. **Methodology**: Main methodological approaches
5. **Results**: Key data results supporting contributions
6. **Advantages**: 3 key strengths of the work
7. **Disadvantages**: 3 limitations or areas for improvement
8. **Conclusion**: 1-2 sentence overall assessment

## Configuration

### Model Options

#### OpenAI Models
- `gpt-4`: More thorough analysis (slower, more expensive)
- `gpt-3.5-turbo`: Faster analysis (cheaper, still effective)
- `gpt-4-turbo`: Latest GPT-4 variant

#### Local LLM Models
- `meta-llama/Llama-3.1-8B-Instruct`: Good balance of quality and speed
- `meta-llama/Llama-3.1-70B-Instruct`: Higher quality (requires more resources)
- Any model supported by your local server

### Temperature Settings
- `0.1`: Conservative, focused analysis (recommended)
- `0.3`: More creative interpretations
- `0.0`: Most deterministic output

### Local LLM Server Requirements
- **Memory**: 8GB+ RAM for 8B models, 40GB+ for 70B models
- **GPU**: Optional but recommended for faster inference
- **Disk**: Model storage space (4-140GB depending on model)
- **Network**: Server must be accessible via HTTP API

## Project Structure

```
paper-review-agent/
├── paper-review.py          # Main agent implementation
├── requirements.txt         # Python dependencies
├── activate.sh             # Virtual environment activation script
├── .gitignore              # Git ignore rules
├── .env                    # API keys (create this)
├── papers/                 # Place PDF files here
├── reviews/                # Generated reviews saved here
└── README.md               # This file
```

## Requirements

- Python 3.8+
- PDF files of academic papers
- **Either**: OpenAI API key (for OpenAI models) **OR** local LLM server (for local models)

## Dependencies

- `langgraph`: Workflow orchestration
- `langchain`: LLM framework
- `langchain-openai`: OpenAI integration (works with local LLM servers too)
- `pypdf`: PDF processing
- `openai`: OpenAI API client
- `requests`: HTTP client for server connectivity testing

## Output

The agent generates a structured markdown review containing:
- Paper title and authors
- Executive summary
- Detailed analysis of contributions, methodology, and results
- Balanced assessment of strengths and weaknesses
- Professional conclusion

### Output Files
Reviews are automatically named based on the PDF and model used:
- OpenAI models: `reviews/paper_name-gpt-4.md`
- Local models: `reviews/paper_name-meta-llama-Llama-3.1-8B-Instruct.md`
- Custom output: Use `--output` to specify your own filename

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Troubleshooting

### Common Issues

1. **Virtual environment not found**
   ```bash
   python3 -m venv paper-review-env
   source paper-review-env/bin/activate
   ```

2. **Missing API key (for OpenAI models)**
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

3. **Local LLM server connection issues**
   - Check if server is running: `curl http://localhost:8080/health` (if health endpoint exists)
   - Verify server URL and port
   - Test with simple request:
     ```bash
     curl -X POST http://localhost:8080/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{"model": "your-model-name", "messages": [{"role": "user", "content": "test"}]}'
     ```
   - Check server logs for errors
   - Ensure model is loaded and accessible

4. **PDF processing errors**
   - Ensure PDF is not password protected
   - Check file path is correct
   - Verify PDF is readable

5. **Dependency issues**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. **Local LLM performance issues**
   - Increase system RAM if available
   - Use smaller model (e.g., 8B instead of 70B)
   - Enable GPU acceleration if available
   - Adjust server configuration for better performance
