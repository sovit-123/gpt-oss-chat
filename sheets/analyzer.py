"""
Sheet Analyzer Module

Provides deep analysis capabilities for spreadsheet data:
- Header analysis and column type detection
- Column importance scoring
- Pattern detection (dates, IDs, categories, numerics)
- Data quality assessment
"""

import pandas as pd
import re
from typing import Optional
from collections import Counter


class SheetAnalyzer:
    """
    Analyzes spreadsheet data to extract insights about structure,
    data types, patterns, and quality.
    """
    
    # Common patterns for column type inference
    ID_PATTERNS = [
        r'^id$', r'.*_id$', r'.*id$', r'^key$', r'.*_key$',
        r'^code$', r'.*_code$', r'^no$', r'.*_no$', r'^number$'
    ]
    
    DATE_PATTERNS = [
        r'date', r'time', r'created', r'updated', r'modified',
        r'timestamp', r'_at$', r'_on$', r'dob', r'birthday'
    ]
    
    AMOUNT_PATTERNS = [
        r'amount', r'price', r'cost', r'total', r'sum', r'value',
        r'revenue', r'profit', r'balance', r'fee', r'salary', r'wage'
    ]
    
    NAME_PATTERNS = [
        r'name', r'title', r'label', r'description', r'desc'
    ]
    
    def __init__(self, reader):
        """
        Initialize the analyzer with a SheetReader instance.
        
        Args:
            reader: SheetReader instance for data access
        """
        self.reader = reader
        self._analysis_cache = {}
    
    def analyze_headers(self, sheet_name: str = 'Sheet1') -> dict:
        """
        Analyze all headers/columns in a sheet.
        
        Args:
            sheet_name: Name of the sheet to analyze
            
        Returns:
            Dictionary with header analysis:
            {
                'headers': [...],
                'column_count': int,
                'inferred_types': {...},
                'key_columns': [...],
                'potential_foreign_keys': [...]
            }
        """
        cache_key = f'headers_{sheet_name}'
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        df = self.reader.read_sheet(sheet_name)
        headers = list(df.columns)
        
        analysis = {
            'headers': headers,
            'column_count': len(headers),
            'inferred_types': {},
            'key_columns': [],
            'potential_foreign_keys': [],
            'column_categories': {
                'identifiers': [],
                'dates': [],
                'amounts': [],
                'names': [],
                'other': []
            }
        }
        
        for col in headers:
            col_type = self._infer_column_type(df, col)
            analysis['inferred_types'][col] = col_type
            
            # Categorize columns
            category = self._categorize_column(col, col_type)
            analysis['column_categories'][category].append(col)
            
            # Check if it might be a key column
            if self._is_potential_key(df, col):
                analysis['key_columns'].append(col)
            
            # Check for foreign key pattern
            if self._is_potential_foreign_key(col):
                analysis['potential_foreign_keys'].append(col)
        
        self._analysis_cache[cache_key] = analysis
        return analysis
    
    def analyze_column(self, sheet_name: str, column: str) -> dict:
        """
        Perform detailed analysis on a specific column.
        
        Args:
            sheet_name: Name of the sheet
            column: Column name to analyze
            
        Returns:
            Detailed column analysis dictionary
        """
        df = self.reader.read_sheet(sheet_name)
        
        if column not in df.columns:
            return {'error': f'Column "{column}" not found'}
        
        series = df[column]
        
        analysis = {
            'name': column,
            'dtype': str(series.dtype),
            'inferred_type': self._infer_column_type(df, column),
            'statistics': self.reader.get_column_stats(sheet_name, column),
            'quality': self._assess_column_quality(series),
            'patterns': self._detect_patterns(series),
            'sample_values': self._get_sample_values(series),
        }
        
        return analysis
    
    def detect_column_types(self, sheet_name: str) -> dict:
        """
        Detect semantic types for all columns (beyond pandas dtypes).
        
        Args:
            sheet_name: Name of the sheet
            
        Returns:
            Dictionary mapping column names to semantic types
        """
        df = self.reader.read_sheet(sheet_name)
        types = {}
        
        for col in df.columns:
            types[col] = self._infer_column_type(df, col)
        
        return types
    
    def find_key_columns(self, sheet_name: str) -> list:
        """
        Identify columns that could serve as primary keys.
        
        Args:
            sheet_name: Name of the sheet
            
        Returns:
            List of potential key column names
        """
        df = self.reader.read_sheet(sheet_name)
        keys = []
        
        for col in df.columns:
            if self._is_potential_key(df, col):
                keys.append({
                    'column': col,
                    'unique_count': int(df[col].nunique()),
                    'total_count': len(df),
                    'is_unique': df[col].nunique() == len(df)
                })
        
        # Sort by uniqueness ratio
        keys.sort(key=lambda x: x['unique_count'] / x['total_count'], reverse=True)
        return keys
    
    def assess_data_quality(self, sheet_name: str) -> dict:
        """
        Assess overall data quality for a sheet.
        
        Args:
            sheet_name: Name of the sheet
            
        Returns:
            Data quality assessment dictionary
        """
        df = self.reader.read_sheet(sheet_name)
        
        total_cells = df.size
        null_cells = df.isna().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Per-column quality
        column_quality = {}
        for col in df.columns:
            column_quality[col] = self._assess_column_quality(df[col])
        
        # Calculate overall score (0-100)
        null_score = max(0, 100 - (null_cells / total_cells * 100))
        duplicate_score = max(0, 100 - (duplicate_rows / len(df) * 100))
        overall_score = (null_score + duplicate_score) / 2
        
        return {
            'overall_score': round(overall_score, 1),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': int(total_cells),
            'null_cells': int(null_cells),
            'null_percentage': round(null_cells / total_cells * 100, 2),
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': round(duplicate_rows / len(df) * 100, 2),
            'column_quality': column_quality,
            'issues': self._identify_quality_issues(df, column_quality)
        }
    
    def get_value_distribution(
        self, 
        sheet_name: str, 
        column: str, 
        top_n: int = 20
    ) -> dict:
        """
        Get value distribution for a column.
        
        Args:
            sheet_name: Name of the sheet
            column: Column name
            top_n: Number of top values to return
            
        Returns:
            Value distribution dictionary
        """
        series = self.reader.get_column_data(sheet_name, column)
        
        value_counts = series.value_counts()
        
        distribution = {
            'total_values': len(series),
            'unique_values': int(series.nunique()),
            'top_values': value_counts.head(top_n).to_dict(),
            'bottom_values': value_counts.tail(5).to_dict() if len(value_counts) > top_n else {},
        }
        
        # Add histogram for numeric data
        if pd.api.types.is_numeric_dtype(series):
            try:
                hist, bins = pd.cut(series.dropna(), bins=10, retbins=True)
                distribution['histogram'] = {
                    'bins': [float(b) for b in bins],
                    'counts': hist.value_counts().sort_index().tolist()
                }
            except Exception:
                pass
        
        return distribution
    
    def _infer_column_type(self, df: pd.DataFrame, column: str) -> str:
        """Infer semantic type for a column."""
        series = df[column]
        col_lower = column.lower()
        dtype = str(series.dtype)
        
        # Check name patterns first
        for pattern in self.ID_PATTERNS:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return 'identifier'
        
        for pattern in self.DATE_PATTERNS:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return 'datetime'
        
        for pattern in self.AMOUNT_PATTERNS:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return 'currency/amount'
        
        for pattern in self.NAME_PATTERNS:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return 'text/name'
        
        # Infer from data type
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        if pd.api.types.is_bool_dtype(series):
            return 'boolean'
        
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() < 10:
                return 'categorical/numeric'
            return 'numeric'
        
        # Check string content
        if dtype == 'object':
            sample = series.dropna().head(100)
            if len(sample) > 0:
                # Check if it looks like dates
                try:
                    pd.to_datetime(sample, errors='raise')
                    return 'datetime'
                except Exception:
                    pass
                
                # Check if it looks like email
                if sample.astype(str).str.contains('@').mean() > 0.5:
                    return 'email'
                
                # Check cardinality for categorical
                if series.nunique() / len(series) < 0.1:
                    return 'categorical'
                
                return 'text'
        
        return 'unknown'
    
    def _categorize_column(self, column: str, col_type: str) -> str:
        """Categorize column into broad categories."""
        if col_type == 'identifier':
            return 'identifiers'
        if col_type == 'datetime':
            return 'dates'
        if col_type in ['currency/amount', 'numeric']:
            return 'amounts'
        if col_type in ['text/name', 'text']:
            return 'names'
        return 'other'
    
    def _is_potential_key(self, df: pd.DataFrame, column: str) -> bool:
        """Check if column could be a primary key."""
        series = df[column]
        col_lower = column.lower()
        
        # Check for ID patterns
        has_id_pattern = any(
            re.search(pattern, col_lower, re.IGNORECASE) 
            for pattern in self.ID_PATTERNS
        )
        
        # Check uniqueness
        uniqueness_ratio = series.nunique() / len(series)
        is_mostly_unique = uniqueness_ratio > 0.9
        
        # Check for nulls
        has_few_nulls = series.isna().sum() / len(series) < 0.01
        
        return (has_id_pattern or is_mostly_unique) and has_few_nulls
    
    def _is_potential_foreign_key(self, column: str) -> bool:
        """Check if column name suggests a foreign key."""
        col_lower = column.lower()
        
        # Foreign keys often end with _id or reference other tables
        fk_patterns = [r'.*_id$', r'^fk_', r'.*_fk$', r'.*_ref$']
        
        return any(
            re.search(pattern, col_lower, re.IGNORECASE) 
            for pattern in fk_patterns
        )
    
    def _assess_column_quality(self, series: pd.Series) -> dict:
        """Assess quality metrics for a single column."""
        total = len(series)
        
        quality = {
            'completeness': round((1 - series.isna().sum() / total) * 100, 1),
            'uniqueness': round(series.nunique() / total * 100, 1),
        }
        
        # Check for potential issues
        issues = []
        
        if series.isna().sum() / total > 0.1:
            issues.append('high_null_rate')
        
        if series.nunique() == 1:
            issues.append('single_value')
        
        if series.nunique() == total and total > 10:
            issues.append('all_unique')
        
        quality['issues'] = issues
        quality['score'] = quality['completeness']
        
        return quality
    
    def _detect_patterns(self, series: pd.Series) -> dict:
        """Detect patterns in column data."""
        patterns = {
            'has_pattern': False,
            'detected_patterns': []
        }
        
        sample = series.dropna().astype(str).head(100)
        
        if len(sample) == 0:
            return patterns
        
        # Check for common patterns
        pattern_checks = [
            ('email', r'^[\w\.-]+@[\w\.-]+\.\w+$'),
            ('phone', r'^[\d\s\-\+\(\)]+$'),
            ('url', r'^https?://'),
            ('uuid', r'^[a-f0-9\-]{36}$'),
            ('date_iso', r'^\d{4}-\d{2}-\d{2}'),
            ('numeric_string', r'^\d+$'),
        ]
        
        for name, pattern in pattern_checks:
            match_rate = sample.str.match(pattern, case=False).mean()
            if match_rate > 0.7:
                patterns['detected_patterns'].append({
                    'type': name,
                    'match_rate': round(match_rate * 100, 1)
                })
                patterns['has_pattern'] = True
        
        return patterns
    
    def _get_sample_values(self, series: pd.Series, n: int = 5) -> list:
        """Get sample values from a column."""
        sample = series.dropna().head(n)
        return [str(v) for v in sample.tolist()]
    
    def _identify_quality_issues(self, df: pd.DataFrame, column_quality: dict) -> list:
        """Identify data quality issues across the sheet."""
        issues = []
        
        # Check for columns with high null rates
        high_null_cols = [
            col for col, q in column_quality.items() 
            if q['completeness'] < 50
        ]
        if high_null_cols:
            issues.append({
                'type': 'high_null_rate',
                'severity': 'warning',
                'columns': high_null_cols,
                'message': f'{len(high_null_cols)} columns have >50% null values'
            })
        
        # Check for single-value columns
        single_value_cols = [
            col for col, q in column_quality.items()
            if 'single_value' in q.get('issues', [])
        ]
        if single_value_cols:
            issues.append({
                'type': 'single_value_columns',
                'severity': 'info',
                'columns': single_value_cols,
                'message': f'{len(single_value_cols)} columns contain only one unique value'
            })
        
        return issues
    
    def get_analysis_summary(self, sheet_name: str) -> str:
        """
        Get a text summary of sheet analysis for LLM consumption.
        
        Args:
            sheet_name: Name of the sheet
            
        Returns:
            Formatted analysis summary string
        """
        headers = self.analyze_headers(sheet_name)
        quality = self.assess_data_quality(sheet_name)
        
        summary_parts = [
            f'=== Analysis Summary for "{sheet_name}" ===',
            '',
            f'Columns: {headers["column_count"]}',
            f'Data Quality Score: {quality["overall_score"]}/100',
            '',
            'Column Categories:',
        ]
        
        for category, cols in headers['column_categories'].items():
            if cols:
                summary_parts.append(f'  {category.title()}: {", ".join(cols)}')
        
        summary_parts.extend([
            '',
            'Potential Key Columns:',
            f'  {", ".join(headers["key_columns"]) if headers["key_columns"] else "None detected"}',
            '',
            'Potential Foreign Keys:',
            f'  {", ".join(headers["potential_foreign_keys"]) if headers["potential_foreign_keys"] else "None detected"}',
        ])
        
        if quality['issues']:
            summary_parts.extend(['', 'Quality Issues:'])
            for issue in quality['issues']:
                summary_parts.append(f'  - {issue["message"]}')
        
        return '\n'.join(summary_parts)
