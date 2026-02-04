# gpt-oss-chat

A simple local RAG + web search pipeline powered by gpt-oss-20b via llama.cpp and similar scale models like NVIDIA Nemotron 3 Nano 30B A3B.

**Terminal chat powered by Rich Console UI**

![](assets/gpt-oss-chat-terminal.png) 

**Gradio chat with a terminal theme**

![](assets/gpt-oss-chat-ui.png)

## Setup Steps

* Install llama.cpp with CUDA

* Install Qdrant docker (optional). Qdrant Python client is mandatory for in memory vector DB and RAG

* Run `pip install -r requirements.txt`

* Create a `.env` file and add the [Tavily](https://www.tavily.com/) API key for web search. Optionally, you can also add the [Perplexity API key](https://docs.perplexity.ai/guides/search-quickstart).

  ```
  TAVILY_API_KEY=YOUR_TAVILY_API_KEY
  PERPLEXITY_API_KEY=YOUR_PERPLEXITY_API_KEY
  ```

## Running

**Start the llama.cpp server:**

```
./build/bin/llama-server -hf ggml-org/gpt-oss-20b-GGUF
```

**Terminal chat**

```
python api_call.py
```

**OR Gradio UI**

```
python app.py
```

---

## 📊 Sheets Agent (NEW!)

An intelligent Excel/CSV analysis agent with human-in-the-loop guidance. The agent can analyze spreadsheets, discover patterns, find relationships between sheets, and generate comprehensive reports.

### Features
- **Multi-format Support**: Excel (.xlsx, .xls) and CSV files
- **Deep Statistical Insights**: Automatically detects correlations, time-series trends, outliers, and categorical dominance (Pareto analysis).
- **Intelligent Analysis**: Automatic column type detection, pattern recognition, and data quality assessment.
- **Relationship Discovery**: Finds connections between sheets through common columns and value overlaps.
- **Human-in-the-Loop**: Interactive prompts and follow-up mode for specific questions ("Show outliers in Sales").
- **Structured Output**: Generates a full analysis suite with master summary, per-sheet insight reports, and JSON hierarchy.

### Usage
**Basic usage:**
```bash
python sheets_agent.py --file /path/to/data.xlsx
```

**Auto mode (rapid insight generation):**
```bash
python sheets_agent.py --file data.csv --auto
```

### Analysis Capability
The agent goes beyond basic row/column counts to find specific business insights:
- **Correlations**: "Strong positive correlation (0.95) between 'Marketing_Spend' and 'Revenue'."
- **Outliers**: "Found 5 extreme outliers in 'Transaction_Value' (values > 3σ)."
- **Dominance**: "Region 'North' accounts for 85% of all orders (Pareto distribution)."
- **Trends**: Detects time-based patterns in numeric data.

### Output Structure
The agent creates a dedicated timestamped folder for each analysis listing:
```
sheets_output/filename_TIMESTAMP/
├── README_Analysis.md      # Master executive summary
├── full_structure.json     # Complete machine-readable hierarchy
└── SheetName_insights.md   # Deep dive for each sheet
```

### Example Report Snippet
```markdown
## Key Insights for "Sales_Q1"
- 🔴 Strong positive correlation (0.92) between **Units** and **Revenue**.
- 🟡 Column **Region** follows a Pareto distribution.
  - *The top 2 values account for 82.0% of all data.*
- 🟡 Column **Discount** has 14 detected outliers.
```

### Requirements

The sheets agent requires additional dependencies (included in requirements.txt):

```
pandas
openpyxl
xlrd
tabulate
```

