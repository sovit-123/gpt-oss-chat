"""
Sheets Agent Tools Module

Defines LLM-callable tools for the sheets agent.
These tools are used by the agent to analyze spreadsheet data.
"""

from typing import Optional


# Global state for the sheets agent context
# This will be set by the main sheets_agent.py
_sheets_context = {
    'reader': None,
    'analyzer': None,
    'connection_finder': None,
    'summarizer': None,
    'user_context': '',
    'focus_sheets': []
}


def set_sheets_context(
    reader=None,
    analyzer=None,
    connection_finder=None,
    summarizer=None,
    user_context: str = '',
    focus_sheets: list = None
):
    """
    Set the global sheets context for tool execution.
    
    Args:
        reader: SheetReader instance
        analyzer: SheetAnalyzer instance
        connection_finder: ConnectionFinder instance
        summarizer: SheetSummarizer instance
        user_context: User-provided context about the data
        focus_sheets: List of sheet names to focus on
    """
    global _sheets_context
    _sheets_context['reader'] = reader
    _sheets_context['analyzer'] = analyzer
    _sheets_context['connection_finder'] = connection_finder
    _sheets_context['summarizer'] = summarizer
    _sheets_context['user_context'] = user_context
    _sheets_context['focus_sheets'] = focus_sheets or []


# Tool definitions for OpenAI-style function calling
sheets_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'list_sheets',
            'description': 'List all sheets/tabs in the workbook with basic info including row count, column count, and headers. Use this first to understand the workbook structure.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'analyze_headers',
            'description': 'Analyze headers and column structure of a specific sheet. Returns column types, key columns, potential foreign keys, and column categories.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'sheet_name': {
                        'type': 'string',
                        'description': 'Name of the sheet to analyze'
                    }
                },
                'required': ['sheet_name']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'analyze_column',
            'description': 'Get detailed statistics and patterns for a specific column including data type, null percentage, unique values, and sample data.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'sheet_name': {
                        'type': 'string',
                        'description': 'Name of the sheet containing the column'
                    },
                    'column_name': {
                        'type': 'string',
                        'description': 'Name of the column to analyze'
                    }
                },
                'required': ['sheet_name', 'column_name']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'sample_data',
            'description': 'Get a sample of rows from a sheet for inspection. Useful for understanding actual data values.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'sheet_name': {
                        'type': 'string',
                        'description': 'Name of the sheet to sample from'
                    },
                    'n_samples': {
                        'type': 'integer',
                        'description': 'Number of sample rows to return (default: 5, max: 20)',
                        'default': 5
                    }
                },
                'required': ['sheet_name']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'find_connections',
            'description': 'Find relationships between sheets based on common columns and value overlaps. Returns potential foreign key relationships.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'assess_quality',
            'description': 'Assess data quality for a sheet including null rates, duplicates, and potential issues.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'sheet_name': {
                        'type': 'string',
                        'description': 'Name of the sheet to assess'
                    }
                },
                'required': ['sheet_name']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'generate_summary',
            'description': 'Generate a comprehensive summary of the analysis. Call this when you have gathered enough information.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'format': {
                        'type': 'string',
                        'enum': ['markdown', 'json', 'both'],
                        'description': 'Output format for the summary',
                        'default': 'both'
                    }
                },
                'required': []
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'request_user_feedback',
            'description': 'Ask the user for clarification or guidance. Use when you need more context about the data or are unsure which analysis direction to take.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'question': {
                        'type': 'string',
                        'description': 'The question to ask the user'
                    }
                },
                'required': ['question']
            }
        }
    }
]


def execute_sheets_tool(tool_name: str, **kwargs) -> str:
    """
    Execute a sheets tool and return the result.
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Tool-specific arguments
        
    Returns:
        String result of the tool execution
    """
    global _sheets_context
    
    reader = _sheets_context['reader']
    analyzer = _sheets_context['analyzer']
    connection_finder = _sheets_context['connection_finder']
    summarizer = _sheets_context['summarizer']
    
    if reader is None:
        return 'Error: Sheets context not initialized. Please load a file first.'
    
    try:
        if tool_name == 'list_sheets':
            return _tool_list_sheets(reader)
        
        elif tool_name == 'analyze_headers':
            sheet_name = kwargs.get('sheet_name')
            if not sheet_name:
                return 'Error: sheet_name is required'
            return _tool_analyze_headers(analyzer, sheet_name)
        
        elif tool_name == 'analyze_column':
            sheet_name = kwargs.get('sheet_name')
            column_name = kwargs.get('column_name')
            if not sheet_name or not column_name:
                return 'Error: sheet_name and column_name are required'
            return _tool_analyze_column(analyzer, sheet_name, column_name)
        
        elif tool_name == 'sample_data':
            sheet_name = kwargs.get('sheet_name')
            n_samples = kwargs.get('n_samples', 5)
            if not sheet_name:
                return 'Error: sheet_name is required'
            return _tool_sample_data(reader, sheet_name, min(n_samples, 20))
        
        elif tool_name == 'find_connections':
            return _tool_find_connections(connection_finder)
        
        elif tool_name == 'assess_quality':
            sheet_name = kwargs.get('sheet_name')
            if not sheet_name:
                return 'Error: sheet_name is required'
            return _tool_assess_quality(analyzer, sheet_name)
        
        elif tool_name == 'generate_summary':
            output_format = kwargs.get('format', 'both')
            return _tool_generate_summary(summarizer, output_format)
        
        elif tool_name == 'request_user_feedback':
            question = kwargs.get('question')
            if not question:
                return 'Error: question is required'
            return _tool_request_user_feedback(question)
        
        else:
            return f'Error: Unknown tool: {tool_name}'
    
    except Exception as e:
        return f'Error executing {tool_name}: {str(e)}'


def _tool_list_sheets(reader) -> str:
    """List all sheets in the workbook."""
    info = reader.get_sheet_info()
    
    result_parts = ['=== Workbook Structure ===', '']
    
    for sheet_name, sheet_info in info.items():
        result_parts.append(f'Sheet: "{sheet_name}"')
        result_parts.append(f'  Rows: {sheet_info.get("rows", 0):,}')
        result_parts.append(f'  Columns: {sheet_info.get("columns", 0)}')
        
        headers = sheet_info.get('headers', [])
        if headers:
            headers_str = ', '.join(headers[:10])
            if len(headers) > 10:
                headers_str += f', ... (+{len(headers) - 10} more)'
            result_parts.append(f'  Headers: {headers_str}')
        
        if sheet_info.get('is_sampled'):
            result_parts.append('  Note: Large file - data will be sampled')
        
        result_parts.append('')
    
    return '\n'.join(result_parts)


def _tool_analyze_headers(analyzer, sheet_name: str) -> str:
    """Analyze headers for a sheet."""
    analysis = analyzer.analyze_headers(sheet_name)
    
    result_parts = [
        f'=== Header Analysis for "{sheet_name}" ===',
        '',
        f'Total columns: {analysis["column_count"]}',
        ''
    ]
    
    # Column categories
    result_parts.append('Column Categories:')
    for category, cols in analysis['column_categories'].items():
        if cols:
            result_parts.append(f'  {category.title()}: {", ".join(cols)}')
    
    result_parts.append('')
    
    # Column types
    result_parts.append('Inferred Types:')
    for col, col_type in analysis['inferred_types'].items():
        result_parts.append(f'  {col}: {col_type}')
    
    result_parts.append('')
    
    # Key columns
    if analysis['key_columns']:
        result_parts.append(f'Potential Key Columns: {", ".join(analysis["key_columns"])}')
    
    if analysis['potential_foreign_keys']:
        result_parts.append(
            f'Potential Foreign Keys: {", ".join(analysis["potential_foreign_keys"])}'
        )
    
    return '\n'.join(result_parts)


def _tool_analyze_column(analyzer, sheet_name: str, column_name: str) -> str:
    """Analyze a specific column."""
    analysis = analyzer.analyze_column(sheet_name, column_name)
    
    if 'error' in analysis:
        return f'Error: {analysis["error"]}'
    
    stats = analysis.get('statistics', {})
    quality = analysis.get('quality', {})
    
    result_parts = [
        f'=== Column Analysis: "{column_name}" ===',
        '',
        f'Data Type: {analysis["dtype"]}',
        f'Inferred Type: {analysis["inferred_type"]}',
        '',
        'Statistics:',
        f'  Total values: {stats.get("count", 0):,}',
        f'  Unique values: {stats.get("unique_count", 0):,}',
        f'  Null percentage: {stats.get("null_percentage", 0)}%',
    ]
    
    # Type-specific stats
    if 'mean' in stats:
        result_parts.extend([
            f'  Min: {stats.get("min")}',
            f'  Max: {stats.get("max")}',
            f'  Mean: {stats.get("mean"):.2f}',
            f'  Median: {stats.get("median"):.2f}',
        ])
    
    if 'top_values' in stats:
        result_parts.append('')
        result_parts.append('Top Values:')
        for val, count in list(stats['top_values'].items())[:5]:
            result_parts.append(f'  "{val}": {count}')
    
    # Sample values
    samples = analysis.get('sample_values', [])
    if samples:
        result_parts.append('')
        result_parts.append(f'Sample Values: {", ".join(samples[:5])}')
    
    # Patterns
    patterns = analysis.get('patterns', {})
    if patterns.get('detected_patterns'):
        result_parts.append('')
        result_parts.append('Detected Patterns:')
        for p in patterns['detected_patterns']:
            result_parts.append(f'  {p["type"]}: {p["match_rate"]}% match')
    
    return '\n'.join(result_parts)


def _tool_sample_data(reader, sheet_name: str, n_samples: int) -> str:
    """Get sample data from a sheet."""
    from sheets.reader import format_dataframe_for_llm
    
    df = reader.sample_sheet(sheet_name, n_samples)
    
    result = f'=== Sample Data from "{sheet_name}" ({len(df)} rows) ===\n\n'
    result += format_dataframe_for_llm(df, max_rows=n_samples)
    
    return result


def _tool_find_connections(connection_finder) -> str:
    """Find connections between sheets."""
    if connection_finder is None:
        return 'Error: Connection finder not available'
    
    return connection_finder.get_connections_summary()


def _tool_assess_quality(analyzer, sheet_name: str) -> str:
    """Assess data quality for a sheet."""
    quality = analyzer.assess_data_quality(sheet_name)
    
    result_parts = [
        f'=== Data Quality Assessment for "{sheet_name}" ===',
        '',
        f'Overall Score: {quality["overall_score"]}/100',
        '',
        f'Total rows: {quality["total_rows"]:,}',
        f'Total columns: {quality["total_columns"]}',
        f'Total cells: {quality["total_cells"]:,}',
        '',
        f'Null cells: {quality["null_cells"]:,} ({quality["null_percentage"]}%)',
        f'Duplicate rows: {quality["duplicate_rows"]:,} ({quality["duplicate_percentage"]}%)',
    ]
    
    if quality['issues']:
        result_parts.append('')
        result_parts.append('Issues Found:')
        for issue in quality['issues']:
            result_parts.append(f'  [{issue["severity"].upper()}] {issue["message"]}')
            if issue.get('columns'):
                result_parts.append(f'    Affected: {", ".join(issue["columns"][:5])}')
    
    return '\n'.join(result_parts)


def _tool_generate_summary(summarizer, output_format: str) -> str:
    """Generate analysis summary."""
    global _sheets_context
    
    focus_sheets = _sheets_context.get('focus_sheets') or None
    
    result_parts = ['=== Summary Generation ===', '']
    
    if output_format in ['markdown', 'both']:
        summary = summarizer.generate_full_summary(focus_sheets)
        result_parts.append('Markdown Summary Generated:')
        result_parts.append('')
        # Include a preview of the summary
        lines = summary.split('\n')[:30]
        result_parts.extend(lines)
        if len(summary.split('\n')) > 30:
            result_parts.append('...(truncated for display)...')
    
    if output_format in ['json', 'both']:
        hierarchy = summarizer.generate_json_hierarchy(focus_sheets)
        result_parts.append('')
        result_parts.append('JSON Hierarchy Generated with:')
        result_parts.append(f'  - {len(hierarchy["sheets"])} sheets')
        result_parts.append(f'  - {len(hierarchy["relationships"])} relationships')
        result_parts.append(f'  - {hierarchy["summary"]["total_rows"]:,} total rows')
    
    result_parts.append('')
    result_parts.append('Summary generation complete. Files can be saved to disk if needed.')
    
    return '\n'.join(result_parts)


def _tool_request_user_feedback(question: str) -> str:
    """Request user feedback - returns a special marker for the agent loop."""
    return f'[USER_FEEDBACK_REQUESTED] {question}'
