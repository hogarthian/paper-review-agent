#!/usr/bin/env python3
"""
Visualize the Academic Paper Analysis Agent workflow.
"""

import os

def visualize_workflow():
    """Generate a visual representation of the workflow."""
    
    try:
        # Try to get the Mermaid representation
        print("🎨 Generating workflow visualization...")
        
        # For visualization, we'll create the workflow structure manually
        # to avoid needing API keys or dependencies
        print("\n📊 Academic Paper Analysis Workflow Steps:")
        print("=" * 50)
        
        workflow_steps = [
            "load_pdf - Load and process PDF content",
            "extract_title_authors - Extract paper title and authors",
            "generate_summary - Generate 3-sentence summary",
            "extract_contributions - Identify major contributions", 
            "analyze_methodology - Summarize methodology",
            "analyze_results - Analyze data results",
            "identify_advantages - Find 3 advantages",
            "identify_disadvantages - Find 3 disadvantages", 
            "write_conclusion - Write 1-2 sentence conclusion",
            "compile_review - Generate final review"
        ]
        
        for i, step in enumerate(workflow_steps, 1):
            print(f"  {i:2d}. {step}")
        

        
        # Create a Mermaid diagram of the workflow
        mermaid_diagram = create_mermaid_diagram()
        
        # Save to file
        with open("workflow_diagram.md", "w") as f:
            f.write("# Academic Paper Analysis Workflow\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_diagram)
            f.write("\n```\n")
        
        print(f"\n✅ Workflow diagram saved to: workflow_diagram.md")
        print("📝 You can view this in any Markdown viewer that supports Mermaid diagrams")
        print("   (GitHub, GitLab, VS Code with Mermaid extension, etc.)")
        
        # Method 3: Try to generate image if graphviz is available
        try:
            save_as_image(mermaid_diagram)
        except Exception as e:
            print(f"\n💡 To generate PNG/SVG images, install: pip install pygraphviz")
            
    except Exception as e:
        print(f"❌ Error generating visualization: {e}")

def create_mermaid_diagram():
    """Create a Mermaid diagram of the workflow."""
    return """graph TD
    A[📄 load_pdf] --> B[📌 extract_title_authors]
    B --> C[📝 generate_summary]
    C --> D[🎯 extract_contributions]
    D --> E[🔬 analyze_methodology]
    E --> F[📊 analyze_results]
    F --> G[✅ identify_advantages]
    G --> H[❌ identify_disadvantages]
    H --> I[🎯 write_conclusion]
    I --> J[📋 compile_review]
    J --> K[🏁 END]
    
    style A fill:#e1f5fe
    style J fill:#e8f5e8
    style K fill:#ffebee"""

def save_as_image(mermaid_diagram):
    """Try to save diagram as image using available tools."""
    try:
        # Try using mermaid-cli if available
        import subprocess
        
        # Save mermaid source
        with open("workflow.mmd", "w") as f:
            f.write(mermaid_diagram)
        
        # Try to convert to PNG
        result = subprocess.run([
            "mmdc", "-i", "workflow.mmd", "-o", "workflow.png"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ PNG image saved to: workflow.png")
        else:
            print(f"💡 To generate images, install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
            
    except FileNotFoundError:
        print(f"💡 mermaid-cli not found. Install with: npm install -g @mermaid-js/mermaid-cli")
    except Exception as e:
        print(f"⚠️  Could not generate image: {e}")

if __name__ == "__main__":
    visualize_workflow() 