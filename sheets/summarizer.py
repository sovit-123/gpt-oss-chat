"""
Sheet Summarizer Module

Generates comprehensive summaries and outputs:
- Markdown report generation
- JSON hierarchy creation
- Executive summary for LLM consumption
- Detailed insight reports in dedicated directories
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import the new Insights module
from sheets.insights import InsightGenerator

class SheetSummarizer:
    """
    Generates summaries and structured outputs from sheet analysis.
    """
    
    def __init__(self, reader, analyzer, connection_finder=None, user_context: str = ''):
        """
        Initialize the summarizer.
        
        Args:
            reader: SheetReader instance
            analyzer: SheetAnalyzer instance
            connection_finder: ConnectionFinder instance (optional)
            user_context: str, optional context provided by the user
        """
        self.reader = reader
        self.analyzer = analyzer
        self.connection_finder = connection_finder
        self.user_context = user_context
        self.insight_generator = InsightGenerator(analyzer)
    
    def generate_detailed_insights(self, sheet_name: str) -> str:
        """
        Generate a text block of deep insights for a sheet.
        """
        df = self.reader.read_sheet(sheet_name)
        insights_list = self.insight_generator.generate_sheet_insights(sheet_name, df)
        
        if not insights_list:
            return "No significant statistical patterns detected."
            
        text = []
        for insight in insights_list:
            icon = "🔴" if insight.get("severity") == "high" else "🟡"
            text.append(f"- {icon} {insight['message']}")
            if 'details' in insight:
                 text.append(f"  - *{insight['details']}*")
        
        return "\n".join(text)

    def generate_sheet_report_markdown(self, sheet_name: str) -> str:
        """
        Generate a FULL markdown report for a SINGLE sheet (to be saved as its own file).
        """
        info = self.reader.get_sheet_info().get(sheet_name, {})
        header_analysis = self.analyzer.analyze_headers(sheet_name)
        quality = self.analyzer.assess_data_quality(sheet_name)
        
        report = [
            f'# Analysis Report: {sheet_name}',
            f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            '',
            '## 1. Overview',
            f'- **Rows**: {info.get("rows", 0):,}',
            f'- **Columns**: {info.get("columns", 0)}',
            f'- **Quality Score**: {quality["overall_score"]}/100',
            '',
            '## 2. Key Insights',
            self.generate_detailed_insights(sheet_name),
            '',
            '## 3. Column Structure',
            '| Column | Inferred Type | Null % | Unique |',
            '|--------|---------------|--------|--------|'
        ]
        
        df = self.reader.read_sheet(sheet_name)
        for col in df.columns:
            col_type = header_analysis['inferred_types'].get(col, 'unknown')
            null_pct = round(df[col].isna().sum() / len(df) * 100, 1)
            unique = df[col].nunique()
            report.append(f'| {col} | {col_type} | {null_pct}% | {unique:,} |')
            
        report.append('')
        report.append('## 4. Data Quality Profile')
        if quality['issues']:
             for issue in quality['issues']:
                report.append(f'- **{issue["type"].replace("_", " ").title()}**: {issue["message"]}')
        else:
            report.append("- No critical quality issues found.")

        return '\n'.join(report)

    def generate_sheet_summary(self, sheet_name: str) -> str:
        """
        Generate a concise summary for the main overview file.
        """
        info = self.reader.get_sheet_info().get(sheet_name, {})
        return f"- **{sheet_name}**: {info.get('rows', 0):,} rows, {info.get('columns', 0)} cols. " \
               f"Detailed analysis in `{sheet_name}_insights.md`."

    def generate_full_summary(self, focus_sheets: Optional[list] = None, llm_summary: str = '') -> str:
        """
        Generate the master README/Summary for the workbook.
        If llm_summary is provided (which is a full report authored by the agent),
        it takes precedence as the main content.
        """
        
        # If we have a comprehensive LLM report, use it as the source of truth
        if llm_summary and len(llm_summary) > 100:
            # Ensure it has a title
            output = []
            if not llm_summary.strip().startswith('#'):
                 output.append(f'# Workbook Analysis: {self.reader.file_path.name}')
                 output.append(f'**Analysis Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
                 output.append('')
            
            output.append(llm_summary)
            
            # Add footer with context if not present
            if self.user_context and 'Context' not in llm_summary:
                 output.append('')
                 output.append('---')
                 output.append('## User Context provided')
                 output.append(f'> {self.user_context}')
                 
            return '\n'.join(output)

        # Fallback to deterministic generation if LLM summary is missing or too short
        sheet_names = focus_sheets or self.reader.get_sheet_names()
        
        summary = [
            f'# Workbook Analysis: {self.reader.file_path.name}',
            f'**Analysis Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            '', 
            '## 1. Project Context',
            f'{self.user_context if self.user_context else "No user context provided."}',
            '',
            '## 2. Sheet Directory',
            f'**Total Sheets**: {len(sheet_names)}',
        ]
        
        for sheet_name in sheet_names:
            summary.append(self.generate_sheet_summary(sheet_name))
            
        if self.connection_finder:
            relationships = self.connection_finder.infer_relationships()
            if relationships:
                summary.append('')
                summary.append('## Entity Relationships')
                for rel in relationships:
                     summary.append(f'- **{rel["from_sheet"]}** -> **{rel["to_sheet"]}** via `{rel["from_column"]}` ({rel["relationship_type"]})')

        return '\n'.join(summary)

    def generate_structured_output(self, output_dir: str, focus_sheets: Optional[list] = None, llm_summary: str = '') -> str:
        """
        Create a directory for this file and fill it with detailed reports.
        """
        sheet_names = focus_sheets or self.reader.get_sheet_names()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create a specific folder for this analysis run
        file_stem = self.reader.file_path.stem
        run_dir = Path(output_dir) / f"{file_stem}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Generate Master Summary with LLM insights
        master_summary = self.generate_full_summary(focus_sheets, llm_summary)
        with open(run_dir / "README_Analysis.md", "w") as f:
            f.write(master_summary)
            
        # 2. Generate Detail File per Sheet
        for sheet in sheet_names:
            detail_md = self.generate_sheet_report_markdown(sheet)
            safe_name = "".join([c if c.isalnum() else "_" for c in sheet])
            with open(run_dir / f"{safe_name}_insights.md", "w") as f:
                f.write(detail_md)
                
        # 3. Generate JSON Hierarchy
        hierarchy = self.generate_json_hierarchy(focus_sheets)
        with open(run_dir / "full_structure.json", "w") as f:
            json.dump(hierarchy, f, indent=2, default=str)
            
        return str(run_dir)
        
    def generate_json_hierarchy(self, focus_sheets: Optional[list] = None) -> dict:
        """Same as before, simplified for this context."""
        sheet_names = focus_sheets or self.reader.get_sheet_names()
        hierarchy = {
            'workbook': str(self.reader.file_path.name),
            'sheets': []
        }
        for sheet_name in sheet_names:
             info = self.reader.get_sheet_info().get(sheet_name, {})
             hierarchy['sheets'].append({
                 'name': sheet_name,
                 'rows': info.get('rows'),
                 'columns': info.get('columns')
             })
        return hierarchy

    # Keep necessary legacy methods for compatibility if needed, 
    # but the above cover the core "proper insights" requirement.
    def get_llm_context(self, focus_sheets: Optional[list] = None) -> str:
         # Simplified context provider
         return self.generate_full_summary(focus_sheets)

    # Re-implement generate_markdown_report to satisfy existing calls in sheets_agent.py
    # But redirect it to use the new structured output logic if possible, 
    # or keep it simple for single-file backwards compatibility
    def generate_markdown_report(self, output_path: str, focus_sheets: Optional[list] = None) -> str:
        content = self.generate_full_summary(focus_sheets)
        with open(output_path, 'w') as f:
            f.write(content)
        return output_path
        
    def generate_json_output(self, output_path: str, focus_sheets: Optional[list] = None) -> str:
        hierarchy = self.generate_json_hierarchy(focus_sheets)
        with open(output_path, 'w') as f:
            json.dump(hierarchy, f, indent=2, default=str)
        return output_path

    def generate_connections_json(self, output_path: str) -> str:
        if self.connection_finder:
            rel = self.connection_finder.generate_relationship_map()
            with open(output_path, 'w') as f:
                 json.dump(rel, f, indent=2, default=str)
        return output_path
