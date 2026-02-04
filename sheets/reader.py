"""
Sheet Reader Module

Handles reading Excel (.xlsx, .xls) and CSV files with support for:
- Multi-sheet detection
- Chunked loading for large files
- Memory-efficient preview generation
- Metadata extraction
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union


class SheetReader:
    """
    Excel/CSV file reader with chunked loading support.
    
    Supports .xlsx, .xls, and .csv formats with intelligent
    handling of large files through chunking.
    """
    
    SUPPORTED_EXTENSIONS = {'.xlsx', '.xls', '.csv'}
    
    def __init__(self, file_path: str, chunk_size: int = 1000, max_rows: int = 5000):
        """
        Initialize the sheet reader.
        
        Args:
            file_path: Path to the Excel or CSV file
            chunk_size: Number of rows per chunk for large files
            max_rows: Maximum rows to load per sheet (for memory efficiency)
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.max_rows = max_rows
        
        if not self.file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')
        
        if self.file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f'Unsupported file format: {self.file_path.suffix}. '
                f'Supported formats: {self.SUPPORTED_EXTENSIONS}'
            )
        
        self._is_csv = self.file_path.suffix.lower() == '.csv'
        self._sheets_cache = {}
        self._info_cache = None
        
    def get_sheet_names(self) -> list:
        """
        Get list of all sheet names in the workbook.
        
        Returns:
            List of sheet names. For CSV files, returns ['Sheet1'].
        """
        if self._is_csv:
            return ['Sheet1']
        
        try:
            xlsx = pd.ExcelFile(self.file_path)
            return xlsx.sheet_names
        except Exception as e:
            raise RuntimeError(f'Error reading sheet names: {e}')
    
    def get_sheet_info(self) -> dict:
        """
        Get metadata about all sheets in the workbook.
        
        Returns:
            Dictionary with sheet info:
            {
                'sheet_name': {
                    'rows': int,
                    'columns': int,
                    'headers': list[str],
                    'is_sampled': bool
                }
            }
        """
        if self._info_cache is not None:
            return self._info_cache
        
        info = {}
        sheet_names = self.get_sheet_names()
        
        for sheet_name in sheet_names:
            try:
                # Read just enough to get info
                if self._is_csv:
                    # For CSV, count rows efficiently
                    df_preview = pd.read_csv(self.file_path, nrows=5)
                    with open(self.file_path, 'r') as f:
                        row_count = sum(1 for _ in f) - 1  # Subtract header
                else:
                    df_preview = pd.read_excel(
                        self.file_path, 
                        sheet_name=sheet_name, 
                        nrows=5
                    )
                    # Get full row count
                    df_full = pd.read_excel(
                        self.file_path, 
                        sheet_name=sheet_name,
                        usecols=[0]  # Only read first column for counting
                    )
                    row_count = len(df_full)
                
                info[sheet_name] = {
                    'rows': row_count,
                    'columns': len(df_preview.columns),
                    'headers': list(df_preview.columns),
                    'is_sampled': row_count > self.max_rows
                }
            except Exception as e:
                info[sheet_name] = {
                    'error': str(e),
                    'rows': 0,
                    'columns': 0,
                    'headers': []
                }
        
        self._info_cache = info
        return info
    
    def preview_sheet(self, sheet_name: str = 'Sheet1', n_rows: int = 10) -> pd.DataFrame:
        """
        Get a preview of the first n rows of a sheet.
        
        Args:
            sheet_name: Name of the sheet to preview
            n_rows: Number of rows to return
            
        Returns:
            DataFrame with the first n rows
        """
        if self._is_csv:
            return pd.read_csv(self.file_path, nrows=n_rows)
        
        return pd.read_excel(self.file_path, sheet_name=sheet_name, nrows=n_rows)
    
    def get_headers(self, sheet_name: str = 'Sheet1') -> list:
        """
        Get column headers for a specific sheet.
        
        Args:
            sheet_name: Name of the sheet
            
        Returns:
            List of column header names
        """
        info = self.get_sheet_info()
        if sheet_name in info:
            return info[sheet_name].get('headers', [])
        return []
    
    def read_sheet(self, sheet_name: str = 'Sheet1') -> pd.DataFrame:
        """
        Read a complete sheet (up to max_rows).
        
        Args:
            sheet_name: Name of the sheet to read
            
        Returns:
            DataFrame with sheet data
        """
        if sheet_name in self._sheets_cache:
            return self._sheets_cache[sheet_name]
        
        if self._is_csv:
            df = pd.read_csv(self.file_path, nrows=self.max_rows)
        else:
            df = pd.read_excel(
                self.file_path, 
                sheet_name=sheet_name, 
                nrows=self.max_rows
            )
        
        self._sheets_cache[sheet_name] = df
        return df
    
    def read_chunk(
        self, 
        sheet_name: str = 'Sheet1', 
        start: int = 0, 
        end: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read a specific chunk of rows from a sheet.
        
        Args:
            sheet_name: Name of the sheet
            start: Starting row index
            end: Ending row index (exclusive)
            
        Returns:
            DataFrame with the specified rows
        """
        if end is None:
            end = start + self.chunk_size
        
        n_rows = end - start
        
        if self._is_csv:
            return pd.read_csv(
                self.file_path, 
                skiprows=range(1, start + 1),  # Skip rows but keep header
                nrows=n_rows
            )
        
        # For Excel, read the full range and slice
        # This is less efficient but necessary for Excel format
        df = pd.read_excel(
            self.file_path, 
            sheet_name=sheet_name,
            skiprows=start,
            nrows=n_rows
        )
        return df
    
    def sample_sheet(
        self, 
        sheet_name: str = 'Sheet1', 
        n_samples: int = 100,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Get a random sample of rows from a sheet.
        
        Args:
            sheet_name: Name of the sheet
            n_samples: Number of random samples to return
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with sampled rows
        """
        df = self.read_sheet(sheet_name)
        
        if len(df) <= n_samples:
            return df
        
        return df.sample(n=n_samples, random_state=random_state)
    
    def get_column_data(
        self, 
        sheet_name: str, 
        column: str,
        max_values: int = 1000
    ) -> pd.Series:
        """
        Get data from a specific column.
        
        Args:
            sheet_name: Name of the sheet
            column: Column name
            max_values: Maximum values to return
            
        Returns:
            Series with column data
        """
        df = self.read_sheet(sheet_name)
        
        if column not in df.columns:
            raise ValueError(f'Column "{column}" not found in sheet "{sheet_name}"')
        
        series = df[column]
        if len(series) > max_values:
            series = series.head(max_values)
        
        return series
    
    def get_column_stats(self, sheet_name: str, column: str) -> dict:
        """
        Get basic statistics for a column.
        
        Args:
            sheet_name: Name of the sheet
            column: Column name
            
        Returns:
            Dictionary with column statistics
        """
        series = self.get_column_data(sheet_name, column)
        
        stats = {
            'name': column,
            'dtype': str(series.dtype),
            'count': len(series),
            'null_count': int(series.isna().sum()),
            'null_percentage': round(series.isna().sum() / len(series) * 100, 2),
            'unique_count': int(series.nunique()),
        }
        
        # Add type-specific stats
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'min': float(series.min()) if not series.isna().all() else None,
                'max': float(series.max()) if not series.isna().all() else None,
                'mean': float(series.mean()) if not series.isna().all() else None,
                'median': float(series.median()) if not series.isna().all() else None,
                'std': float(series.std()) if not series.isna().all() else None,
            })
        elif pd.api.types.is_datetime64_any_dtype(series):
            stats.update({
                'min_date': str(series.min()) if not series.isna().all() else None,
                'max_date': str(series.max()) if not series.isna().all() else None,
            })
        else:
            # String/categorical
            if stats['unique_count'] <= 20:
                stats['top_values'] = series.value_counts().head(10).to_dict()
            stats['avg_length'] = round(series.dropna().astype(str).str.len().mean(), 2)
        
        return stats
    
    def get_summary_stats(self, sheet_name: str = 'Sheet1') -> str:
        """
        Get a text summary of the sheet for LLM consumption.
        
        Args:
            sheet_name: Name of the sheet
            
        Returns:
            Formatted string with sheet summary
        """
        info = self.get_sheet_info()
        sheet_info = info.get(sheet_name, {})
        
        if 'error' in sheet_info:
            return f'Error reading sheet {sheet_name}: {sheet_info["error"]}'
        
        df = self.read_sheet(sheet_name)
        
        summary_parts = [
            f'Sheet: {sheet_name}',
            f'Rows: {sheet_info["rows"]:,}',
            f'Columns: {sheet_info["columns"]}',
            '',
            'Headers:',
        ]
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_pct = round(df[col].isna().sum() / len(df) * 100, 1)
            summary_parts.append(f'  - {col} ({dtype}, {null_pct}% null)')
        
        return '\n'.join(summary_parts)


def format_dataframe_for_llm(df: pd.DataFrame, max_rows: int = 10) -> str:
    """
    Format a DataFrame as a string for LLM consumption.
    
    Args:
        df: DataFrame to format
        max_rows: Maximum rows to include
        
    Returns:
        Formatted string representation
    """
    if len(df) > max_rows:
        df = df.head(max_rows)
    
    return df.to_markdown(index=False)
