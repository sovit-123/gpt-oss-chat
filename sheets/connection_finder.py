"""
Connection Finder Module

Detects relationships and connections between sheets:
- Foreign key detection
- Common value analysis
- Relationship type inference (one-to-one, one-to-many, many-to-many)
"""

import pandas as pd
from typing import Optional


class ConnectionFinder:
    """
    Discovers relationships between sheets in a workbook
    by analyzing column names and value overlaps.
    """
    
    def __init__(self, reader):
        """
        Initialize the connection finder.
        
        Args:
            reader: SheetReader instance for data access
        """
        self.reader = reader
        self._connections_cache = None
    
    def find_common_columns(self) -> dict:
        """
        Find columns with matching names across sheets.
        
        Returns:
            Dictionary mapping column names to list of sheets containing them.
            {
                'column_name': ['Sheet1', 'Sheet2'],
                ...
            }
        """
        sheet_names = self.reader.get_sheet_names()
        
        if len(sheet_names) < 2:
            return {}
        
        # Collect all columns per sheet
        sheet_columns = {}
        for sheet_name in sheet_names:
            headers = self.reader.get_headers(sheet_name)
            sheet_columns[sheet_name] = set(h.lower() for h in headers)
        
        # Find columns that appear in multiple sheets
        all_columns = set()
        for cols in sheet_columns.values():
            all_columns.update(cols)
        
        common_columns = {}
        for col in all_columns:
            sheets_with_col = [
                sheet for sheet, cols in sheet_columns.items() 
                if col in cols
            ]
            if len(sheets_with_col) > 1:
                common_columns[col] = sheets_with_col
        
        return common_columns
    
    def find_value_overlaps(
        self, 
        sheet1: str, 
        col1: str, 
        sheet2: str, 
        col2: str,
        sample_size: int = 1000
    ) -> dict:
        """
        Find overlapping values between two columns in different sheets.
        
        Args:
            sheet1: First sheet name
            col1: Column name in first sheet
            sheet2: Second sheet name
            col2: Column name in second sheet
            sample_size: Maximum values to compare
            
        Returns:
            Dictionary with overlap analysis:
            {
                'match_count': int,
                'match_percentage_sheet1': float,
                'match_percentage_sheet2': float,
                'sample_matches': list
            }
        """
        try:
            values1 = set(
                self.reader.get_column_data(sheet1, col1, sample_size)
                .dropna()
                .astype(str)
                .tolist()
            )
            values2 = set(
                self.reader.get_column_data(sheet2, col2, sample_size)
                .dropna()
                .astype(str)
                .tolist()
            )
        except (ValueError, KeyError) as e:
            return {'error': str(e)}
        
        overlap = values1 & values2
        
        return {
            'sheet1': sheet1,
            'column1': col1,
            'unique_values1': len(values1),
            'sheet2': sheet2,
            'column2': col2,
            'unique_values2': len(values2),
            'match_count': len(overlap),
            'match_percentage_sheet1': round(len(overlap) / len(values1) * 100, 2) if values1 else 0,
            'match_percentage_sheet2': round(len(overlap) / len(values2) * 100, 2) if values2 else 0,
            'sample_matches': list(overlap)[:10]
        }
    
    def infer_relationships(self, min_overlap_pct: float = 50.0) -> list:
        """
        Automatically infer relationships between sheets.
        
        Args:
            min_overlap_pct: Minimum overlap percentage to consider as relationship
            
        Returns:
            List of discovered relationships:
            [
                {
                    'from_sheet': str,
                    'from_column': str,
                    'to_sheet': str,
                    'to_column': str,
                    'relationship_type': str,
                    'confidence': float,
                    'overlap_info': dict
                }
            ]
        """
        if self._connections_cache is not None:
            return self._connections_cache
        
        sheet_names = self.reader.get_sheet_names()
        
        if len(sheet_names) < 2:
            return []
        
        relationships = []
        common_cols = self.find_common_columns()
        
        # Check relationships based on common column names
        for col_name, sheets in common_cols.items():
            for i, sheet1 in enumerate(sheets):
                for sheet2 in sheets[i+1:]:
                    # Find actual column names (case-insensitive match)
                    headers1 = self.reader.get_headers(sheet1)
                    headers2 = self.reader.get_headers(sheet2)
                    
                    col1 = next((h for h in headers1 if h.lower() == col_name), None)
                    col2 = next((h for h in headers2 if h.lower() == col_name), None)
                    
                    if col1 and col2:
                        overlap = self.find_value_overlaps(sheet1, col1, sheet2, col2)
                        
                        if 'error' not in overlap:
                            max_overlap = max(
                                overlap['match_percentage_sheet1'],
                                overlap['match_percentage_sheet2']
                            )
                            
                            if max_overlap >= min_overlap_pct:
                                rel_type = self._determine_relationship_type(overlap)
                                
                                relationships.append({
                                    'from_sheet': sheet1,
                                    'from_column': col1,
                                    'to_sheet': sheet2,
                                    'to_column': col2,
                                    'relationship_type': rel_type,
                                    'confidence': round(max_overlap, 1),
                                    'overlap_info': overlap
                                })
        
        # Also check for ID columns that might link sheets
        relationships.extend(self._find_id_based_relationships(min_overlap_pct))
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_relationships = []
        for rel in relationships:
            key = frozenset([
                (rel['from_sheet'], rel['from_column']),
                (rel['to_sheet'], rel['to_column'])
            ])
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        unique_relationships.sort(key=lambda x: x['confidence'], reverse=True)
        self._connections_cache = unique_relationships
        
        return unique_relationships
    
    def generate_relationship_map(self) -> dict:
        """
        Generate a complete relationship map of the workbook.
        
        Returns:
            Dictionary describing the workbook structure and relationships.
        """
        sheet_names = self.reader.get_sheet_names()
        sheet_info = self.reader.get_sheet_info()
        relationships = self.infer_relationships()
        
        # Build node list (sheets)
        nodes = []
        for name in sheet_names:
            info = sheet_info.get(name, {})
            nodes.append({
                'id': name,
                'rows': info.get('rows', 0),
                'columns': info.get('columns', 0),
                'headers': info.get('headers', [])
            })
        
        # Build edge list (relationships)
        edges = []
        for rel in relationships:
            edges.append({
                'source': rel['from_sheet'],
                'source_column': rel['from_column'],
                'target': rel['to_sheet'],
                'target_column': rel['to_column'],
                'type': rel['relationship_type'],
                'confidence': rel['confidence']
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'total_sheets': len(nodes),
            'total_relationships': len(edges)
        }
    
    def _determine_relationship_type(self, overlap: dict) -> str:
        """Determine the type of relationship based on overlap analysis."""
        unique1 = overlap['unique_values1']
        unique2 = overlap['unique_values2']
        match = overlap['match_count']
        
        # Check cardinality
        ratio = unique1 / unique2 if unique2 > 0 else 0
        
        if 0.9 <= ratio <= 1.1 and match > min(unique1, unique2) * 0.8:
            return 'one-to-one'
        elif ratio < 0.5:
            return 'many-to-one'
        elif ratio > 2:
            return 'one-to-many'
        else:
            return 'many-to-many'
    
    def _find_id_based_relationships(self, min_overlap_pct: float) -> list:
        """Find relationships based on ID column patterns."""
        sheet_names = self.reader.get_sheet_names()
        relationships = []
        
        # Collect all potential ID columns per sheet
        id_columns = {}
        for sheet_name in sheet_names:
            headers = self.reader.get_headers(sheet_name)
            ids = [h for h in headers if self._is_id_column(h)]
            if ids:
                id_columns[sheet_name] = ids
        
        # Check for cross-sheet ID matches
        sheets = list(id_columns.keys())
        for i, sheet1 in enumerate(sheets):
            for sheet2 in sheets[i+1:]:
                for col1 in id_columns[sheet1]:
                    for col2 in id_columns[sheet2]:
                        # Skip if already checked via common columns
                        if col1.lower() == col2.lower():
                            continue
                        
                        # Check if column names suggest a relationship
                        # e.g., customer_id in Orders referencing id in Customers
                        if self._columns_might_relate(col1, col2, sheet1, sheet2):
                            overlap = self.find_value_overlaps(
                                sheet1, col1, sheet2, col2
                            )
                            
                            if 'error' not in overlap:
                                max_overlap = max(
                                    overlap['match_percentage_sheet1'],
                                    overlap['match_percentage_sheet2']
                                )
                                
                                if max_overlap >= min_overlap_pct:
                                    rel_type = self._determine_relationship_type(overlap)
                                    
                                    relationships.append({
                                        'from_sheet': sheet1,
                                        'from_column': col1,
                                        'to_sheet': sheet2,
                                        'to_column': col2,
                                        'relationship_type': rel_type,
                                        'confidence': round(max_overlap * 0.9, 1),
                                        'overlap_info': overlap,
                                        'inferred': True
                                    })
        
        return relationships
    
    def _is_id_column(self, column: str) -> bool:
        """Check if column name suggests an ID/key column."""
        col_lower = column.lower()
        id_patterns = ['id', 'key', 'code', 'no', 'number', 'ref']
        return any(p in col_lower for p in id_patterns)
    
    def _columns_might_relate(
        self, 
        col1: str, 
        col2: str, 
        sheet1: str, 
        sheet2: str
    ) -> bool:
        """Check if column names suggest a relationship."""
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        sheet1_lower = sheet1.lower()
        sheet2_lower = sheet2.lower()
        
        # Check if col1 references sheet2 (e.g., customer_id in Orders -> Customers)
        if any(word in col1_lower for word in sheet2_lower.split()):
            return True
        
        # Check if col2 references sheet1
        if any(word in col2_lower for word in sheet1_lower.split()):
            return True
        
        # Check for common patterns
        if col1_lower.replace('_id', '') == col2_lower.replace('_id', ''):
            return True
        
        return False
    
    def get_connections_summary(self) -> str:
        """
        Get a text summary of discovered connections for LLM consumption.
        
        Returns:
            Formatted connections summary string
        """
        relationships = self.infer_relationships()
        
        if not relationships:
            return 'No relationships discovered between sheets.'
        
        summary_parts = [
            '=== Sheet Relationships ===',
            '',
            f'Total relationships found: {len(relationships)}',
            ''
        ]
        
        for rel in relationships:
            summary_parts.append(
                f'• {rel["from_sheet"]}.{rel["from_column"]} → '
                f'{rel["to_sheet"]}.{rel["to_column"]}'
            )
            summary_parts.append(
                f'  Type: {rel["relationship_type"]}, '
                f'Confidence: {rel["confidence"]}%'
            )
            summary_parts.append('')
        
        return '\n'.join(summary_parts)
