"""
Sheet Summarizer Module

Generates comprehensive summaries and outputs:
- Markdown report generation
- JSON hierarchy creation
- Executive summary for LLM consumption
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class SheetSummarizer:
    """
    Generates summaries and structured outputs from sheet analysis.
    """
    
    def __init__(self, reader, analyzer, connection_finder=None):
        """
        Initialize the summarizer.
        
        Args:
            reader: SheetReader instance
            analyzer: SheetAnalyzer instance
            connection_finder: ConnectionFinder instance (optional)
        """
        self.reader = reader
        self.analyzer = analyzer
        self.connection_finder = connection_finder
    
    def generate_sheet_summary(self, sheet_name: str) -> str:
        """
        Generate a detailed text summary for a single sheet.
        
        Args:
            sheet_name: Name of the sheet
            
        Returns:
            Markdown-formatted summary string
        """
        info = self.reader.get_sheet_info().get(sheet_name, {})
        header_analysis = self.analyzer.analyze_headers(sheet_name)
        quality = self.analyzer.assess_data_quality(sheet_name)
        
        summary = [
            f'## Sheet: "{sheet_name}"',
            '',
            f'**Rows**: {info.get("rows", 0):,} | '
            f'**Columns**: {info.get("columns", 0)} | '
            f'**Quality Score**: {quality["overall_score"]}/100',
            ''
        ]
        
        # Key columns
        if header_analysis['key_columns']:
            summary.append(
                f'**Key Columns**: {", ".join(header_analysis["key_columns"])}'
            )
        
        # Foreign keys
        if header_analysis['potential_foreign_keys']:
            summary.append(
                f'**Potential Foreign Keys**: '
                f'{", ".join(header_analysis["potential_foreign_keys"])}'
            )
        
        summary.append('')
        summary.append('### Column Details')
        summary.append('')
        summary.append('| Column | Type | Null % | Unique |')
        summary.append('|--------|------|--------|--------|')
        
        df = self.reader.read_sheet(sheet_name)
        for col in df.columns:
            col_type = header_analysis['inferred_types'].get(col, 'unknown')
            null_pct = round(df[col].isna().sum() / len(df) * 100, 1)
            unique = df[col].nunique()
            summary.append(f'| {col} | {col_type} | {null_pct}% | {unique:,} |')
        
        # Quality issues
        if quality['issues']:
            summary.append('')
            summary.append('### Data Quality Issues')
            summary.append('')
            for issue in quality['issues']:
                summary.append(f'- ⚠️ {issue["message"]}')
        
        return '\n'.join(summary)
    
    def generate_full_summary(self, focus_sheets: Optional[list] = None) -> str:
        """
        Generate a complete summary for the entire workbook.
        
        Args:
            focus_sheets: Optional list of sheets to focus on
            
        Returns:
            Complete markdown summary
        """
        sheet_names = focus_sheets or self.reader.get_sheet_names()
        info = self.reader.get_sheet_info()
        
        summary = [
            f'# Workbook Analysis Report',
            '',
            f'**File**: {self.reader.file_path.name}',
            f'**Analysis Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'**Total Sheets**: {len(sheet_names)}',
            '',
            '---',
            '',
            '## Executive Summary',
            '',
        ]
        
        # Quick overview
        total_rows = sum(info.get(s, {}).get('rows', 0) for s in sheet_names)
        total_cols = sum(info.get(s, {}).get('columns', 0) for s in sheet_names)
        
        summary.append(
            f'This workbook contains **{len(sheet_names)} sheets** with a total of '
            f'**{total_rows:,} rows** and **{total_cols} columns**.'
        )
        summary.append('')
        
        # Sheet overview table
        summary.append('### Sheet Overview')
        summary.append('')
        summary.append('| Sheet | Rows | Columns | Quality |')
        summary.append('|-------|------|---------|---------|')
        
        for sheet_name in sheet_names:
            sheet_info = info.get(sheet_name, {})
            try:
                quality = self.analyzer.assess_data_quality(sheet_name)
                quality_score = quality['overall_score']
            except Exception:
                quality_score = 'N/A'
            
            summary.append(
                f'| {sheet_name} | {sheet_info.get("rows", 0):,} | '
                f'{sheet_info.get("columns", 0)} | {quality_score} |'
            )
        
        summary.append('')
        summary.append('---')
        summary.append('')
        
        # Detailed analysis per sheet
        for sheet_name in sheet_names:
            summary.append(self.generate_sheet_summary(sheet_name))
            summary.append('')
            summary.append('---')
            summary.append('')
        
        # Relationships
        if self.connection_finder:
            summary.append('## Relationships')
            summary.append('')
            
            relationships = self.connection_finder.infer_relationships()
            
            if relationships:
                summary.append('The following relationships were discovered between sheets:')
                summary.append('')
                
                for rel in relationships:
                    summary.append(
                        f'- **{rel["from_sheet"]}.{rel["from_column"]}** → '
                        f'**{rel["to_sheet"]}.{rel["to_column"]}** '
                        f'({rel["relationship_type"]}, {rel["confidence"]}% confidence)'
                    )
            else:
                summary.append('No relationships were discovered between sheets.')
        
        return '\n'.join(summary)
    
    def generate_json_hierarchy(self, focus_sheets: Optional[list] = None) -> dict:
        """
        Generate a structured JSON representation of the workbook.
        
        Args:
            focus_sheets: Optional list of sheets to include
            
        Returns:
            Dictionary with complete workbook structure
        """
        sheet_names = focus_sheets or self.reader.get_sheet_names()
        info = self.reader.get_sheet_info()
        
        hierarchy = {
            'workbook': str(self.reader.file_path.name),
            'analysis_date': datetime.now().isoformat(),
            'sheets': [],
            'relationships': [],
            'summary': {}
        }
        
        # Collect sheet details
        for sheet_name in sheet_names:
            sheet_info = info.get(sheet_name, {})
            header_analysis = self.analyzer.analyze_headers(sheet_name)
            
            try:
                quality = self.analyzer.assess_data_quality(sheet_name)
            except Exception:
                quality = {'overall_score': 0}
            
            sheet_data = {
                'name': sheet_name,
                'rows': sheet_info.get('rows', 0),
                'columns': sheet_info.get('columns', 0),
                'headers': sheet_info.get('headers', []),
                'is_sampled': sheet_info.get('is_sampled', False),
                'key_columns': header_analysis.get('key_columns', []),
                'foreign_keys': header_analysis.get('potential_foreign_keys', []),
                'column_types': header_analysis.get('inferred_types', {}),
                'column_categories': header_analysis.get('column_categories', {}),
                'quality_score': quality.get('overall_score', 0)
            }
            
            hierarchy['sheets'].append(sheet_data)
        
        # Add relationships
        if self.connection_finder:
            relationships = self.connection_finder.infer_relationships()
            
            for rel in relationships:
                hierarchy['relationships'].append({
                    'from_sheet': rel['from_sheet'],
                    'from_column': rel['from_column'],
                    'to_sheet': rel['to_sheet'],
                    'to_column': rel['to_column'],
                    'type': rel['relationship_type'],
                    'confidence': rel['confidence']
                })
        
        # Add summary stats
        hierarchy['summary'] = {
            'total_sheets': len(sheet_names),
            'total_rows': sum(s['rows'] for s in hierarchy['sheets']),
            'total_columns': sum(s['columns'] for s in hierarchy['sheets']),
            'total_relationships': len(hierarchy['relationships'])
        }
        
        return hierarchy
    
    def generate_markdown_report(
        self, 
        output_path: str,
        focus_sheets: Optional[list] = None
    ) -> str:
        """
        Generate and save a markdown report.
        
        Args:
            output_path: Path to save the report
            focus_sheets: Optional list of sheets to include
            
        Returns:
            Path to the saved file
        """
        content = self.generate_full_summary(focus_sheets)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_path)
    
    def generate_json_output(
        self, 
        output_path: str,
        focus_sheets: Optional[list] = None
    ) -> str:
        """
        Generate and save the JSON hierarchy.
        
        Args:
            output_path: Path to save the JSON file
            focus_sheets: Optional list of sheets to include
            
        Returns:
            Path to the saved file
        """
        hierarchy = self.generate_json_hierarchy(focus_sheets)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(hierarchy, f, indent=2, default=str)
        
        return str(output_path)
    
    def generate_connections_json(
        self, 
        output_path: str
    ) -> str:
        """
        Generate and save the connections/relationship map.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            Path to the saved file
        """
        if not self.connection_finder:
            raise ValueError('ConnectionFinder not provided')
        
        rel_map = self.connection_finder.generate_relationship_map()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rel_map, f, indent=2, default=str)
        
        return str(output_path)
    
    def get_llm_context(self, focus_sheets: Optional[list] = None) -> str:
        """
        Generate a concise context string for LLM consumption.
        
        This provides enough information for the LLM to understand
        the workbook structure without overwhelming token limits.
        
        Args:
            focus_sheets: Optional list of sheets to include
            
        Returns:
            Concise context string
        """
        sheet_names = focus_sheets or self.reader.get_sheet_names()
        info = self.reader.get_sheet_info()
        
        context_parts = [
            f'Workbook: {self.reader.file_path.name}',
            f'Sheets: {len(sheet_names)}',
            ''
        ]
        
        for sheet_name in sheet_names:
            sheet_info = info.get(sheet_name, {})
            headers = sheet_info.get('headers', [])[:15]  # Limit headers
            
            context_parts.append(f'Sheet "{sheet_name}":')
            context_parts.append(f'  Rows: {sheet_info.get("rows", 0):,}')
            context_parts.append(f'  Columns: {", ".join(headers)}')
            
            if len(sheet_info.get('headers', [])) > 15:
                context_parts.append(
                    f'  ... and {len(sheet_info["headers"]) - 15} more columns'
                )
            context_parts.append('')
        
        if self.connection_finder:
            relationships = self.connection_finder.infer_relationships()
            if relationships:
                context_parts.append('Relationships:')
                for rel in relationships[:5]:  # Limit relationships
                    context_parts.append(
                        f'  {rel["from_sheet"]}.{rel["from_column"]} -> '
                        f'{rel["to_sheet"]}.{rel["to_column"]}'
                    )
        
        return '\n'.join(context_parts)
