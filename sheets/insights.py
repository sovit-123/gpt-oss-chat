"""
Insights Generation Module

Goes beyond basic statistics to find meaningful patterns:
- Correlation analysis
- Time series trends
- Outlier detection
- Pareto/distribution analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class InsightGenerator:
    """
    Generates semantic insights from dataframe data.
    """
    
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def generate_sheet_insights(self, sheet_name: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate a list of interesting insights for a whole sheet.
        """
        insights = []
        
        # 1. Correlation Analysis (Numerical)
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_insights = self._analyze_correlations(numeric_df)
            insights.extend(corr_insights)

        # 2. Time Series / Trends
        date_cols = [
            col for col in df.columns 
            if pd.api.types.is_datetime64_any_dtype(df[col]) 
            or 'date' in col.lower()
        ]
        
        display_date_col = None
        # If we have a date column, let's analyze trends in numeric columns against it
        if date_cols:
            # Try to convert first date-like column to actual datetime if not already
            date_col = date_cols[0]
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    temp_dates = pd.to_datetime(df[date_col], errors='coerce')
                    if temp_dates.notna().sum() > len(df) * 0.5: # If mostly valid dates
                         display_date_col = date_col
                else:
                     display_date_col = date_col
            except:
                pass

        if display_date_col:
            # Group by month/year and look for trends in numeric cols
            pass # TODO: Implement time-series grouping

        # 3. Categorical Dominance (Pareto)
        cat_df = df.select_dtypes(include=['object', 'category'])
        for col in cat_df.columns:
            if df[col].nunique() < 50: # Only analyze manageable categories
                cat_insight = self._analyze_categorical_balance(df, col)
                if cat_insight:
                    insights.append(cat_insight)

        # 4. Outlier Analysis
        for col in numeric_df.columns:
            outlier_insight = self._find_outliers(df, col)
            if outlier_insight:
                insights.append(outlier_insight)

        return insights

    def _analyze_correlations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find strong correlations between columns."""
        insights = []
        corr_matrix = df.corr()
        
        # Iterate over upper triangle
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                val = corr_matrix.iloc[i, j]
                
                if abs(val) > 0.7: # Strong correlation threshold
                    relationship = "positive" if val > 0 else "negative"
                    insights.append({
                        "type": "correlation",
                        "severity": "high" if abs(val) > 0.9 else "medium",
                        "message": f"Strong {relationship} correlation ({val:.2f}) between **{col1}** and **{col2}**.",
                        "details": f"As {col1} increases, {col2} tends to {'increase' if val > 0 else 'decrease'}."
                    })
        return insights

    def _analyze_categorical_balance(self, df: pd.DataFrame, col: str) -> Optional[Dict[str, Any]]:
        """Check for 80/20 rule or single dominator."""
        counts = df[col].value_counts(normalize=True)
        if len(counts) == 0:
            return None
            
        top_val = counts.index[0]
        top_pct = counts.iloc[0]
        
        if top_pct > 0.90:
            return {
                "type": "dominance",
                "severity": "high",
                "message": f"Column **{col}** is dominated by value '{top_val}' ({top_pct:.1%} of rows).",
                "details": "This column implies low variance and might not be useful for differentiation."
            }
        
        if len(counts) > 5:
            # Check Pareto: Do top 20% of categories account for >80% of data?
            cumulative = counts.cumsum()
            twenty_pct_idx = int(len(counts) * 0.2)
            if twenty_pct_idx >= 1:
                top_20_share = cumulative.iloc[twenty_pct_idx - 1]
                if top_20_share > 0.8:
                    return {
                        "type": "pareto",
                        "severity": "medium",
                        "message": f"Column **{col}** follows a Pareto distribution.",
                        "details": f"The top {twenty_pct_idx} values account for {top_20_share:.1%} of all data."
                    }
        return None

    def _find_outliers(self, df: pd.DataFrame, col: str) -> Optional[Dict[str, Any]]:
        """Find meaningful outliers using IQR."""
        series = df[col].dropna()
        if len(series) < 10:
            return None
            
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        if len(outliers) > 0 and len(outliers) / len(series) < 0.05: # Only if they are rare (<5%)
            min_out = outliers.min()
            max_out = outliers.max()
            return {
                "type": "outliers",
                "severity": "medium",
                "message": f"Column **{col}** has {len(outliers)} detected outliers.",
                "details": f"Values range from {min_out} to {max_out}, outside normal range [{lower_bound:.2f}, {upper_bound:.2f}]."
            }
        return None
