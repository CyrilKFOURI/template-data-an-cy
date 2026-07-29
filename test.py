import os
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.feature_selection import mutual_info_regression

# Suppress warnings for clean execution
warnings.filterwarnings('ignore')

def to_markdown_table(df):
    """Formats a pandas DataFrame as a clean Markdown table without external dependencies."""
    if df.empty:
        return "Empty DataFrame"
    df_copy = df.copy()
    # If the index is named or is not a simple range, include it in the table
    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        df_copy = df_copy.reset_index()
    
    # Convert all columns to string, format floats nicely
    for col in df_copy.columns:
        if pd.api.types.is_float_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
        else:
            df_copy[col] = df_copy[col].astype(str)
            
    headers = list(df_copy.columns)
    rows = df_copy.values.tolist()
    
    # Calculate column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(str(val)))
            
    header_line = "| " + " | ".join(f"{str(h):<{widths[idx]}}" for idx, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * w for w in widths) + " |"
    row_lines = ["| " + " | ".join(f"{str(val):<{widths[idx]}}" for idx, val in enumerate(row)) + " |" for row in rows]
    
    return "\n".join([header_line, separator_line] + row_lines)


def detect_datetime_column(series):
    """Heuristic to detect if a column is a datetime, even if encoded as string."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    
    # Sample non-null values
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return False
        
    # Check if they have separators like -, /, :, space
    date_separators = ['-', '/', ':', ' ']
    sample_str = sample.astype(str)
    has_separator = sample_str.apply(lambda x: any(sep in x for sep in date_separators)).mean() > 0.8
    if not has_separator:
        return False
        
    try:
        parsed = pd.to_datetime(sample, errors='coerce')
        return parsed.notna().mean() > 0.8
    except Exception:
        return False


def cramers_v(x, y):
    """Computes Cramér's V statistic for categorical association, ranging from 0 to 1."""
    mask = ~(x.isna() | y.isna())
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return 0.0
    confusion_matrix = pd.crosstab(x, y)
    r, k = confusion_matrix.shape
    if r <= 1 or k <= 1:
        return 0.0
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = len(x)
    return np.sqrt(chi2 / (n * min(r - 1, k - 1)))


def binned_nmi(x, y, bins=10):
    """Computes Normalized Mutual Information (NMI) using quantile binning for continuous variables."""
    mask = ~(x.isna() | y.isna())
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return 0.0
        
    # Bin continuous variable x
    if pd.api.types.is_numeric_dtype(x):
        try:
            x_binned = pd.qcut(x, q=bins, labels=False, duplicates='drop')
        except Exception:
            x_binned = pd.cut(x, bins=bins, labels=False)
    else:
        x_binned = x.astype(str)
        
    # Bin continuous variable y
    if pd.api.types.is_numeric_dtype(y):
        try:
            y_binned = pd.qcut(y, q=bins, labels=False, duplicates='drop')
        except Exception:
            y_binned = pd.cut(y, bins=bins, labels=False)
    else:
        y_binned = y.astype(str)
        
    contingency = pd.crosstab(x_binned, y_binned, normalize=True)
    pi_ = contingency.sum(axis=1).values
    pj_ = contingency.sum(axis=0).values
    pij = contingency.values
    
    # Entropy calculations
    pi_nonzero = pi_[pi_ > 0]
    pj_nonzero = pj_[pj_ > 0]
    h_x = -np.sum(pi_nonzero * np.log(pi_nonzero))
    h_y = -np.sum(pj_nonzero * np.log(pj_nonzero))
    
    # Check for near-zero entropy (constant values)
    if h_x < 1e-10 or h_y < 1e-10:
        return 0.0
        
    # Mutual Information
    mi = 0.0
    for i in range(len(pi_)):
        for j in range(len(pj_)):
            if pij[i, j] > 0:
                mi += pij[i, j] * np.log(pij[i, j] / (pi_[i] * pj_[j]))
                
    nmi = mi / np.sqrt(h_x * h_y)
    return max(0.0, min(1.0, nmi))


def make_psd(matrix, epsilon=1e-6):
    """Projects a symmetric matrix to the nearest positive semi-definite correlation matrix."""
    # Symmetrize
    matrix = (matrix + matrix.T) / 2
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Clip negative eigenvalues
    eigenvalues = np.clip(eigenvalues, epsilon, None)
    # Reconstruct
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Normalize diagonal to be 1.0 (correlation matrix property)
    diag = np.diag(psd_matrix)
    std_dev = np.sqrt(diag)
    psd_matrix = psd_matrix / np.outer(std_dev, std_dev)
    return psd_matrix


class DataProfiler:
    """Analyzes, profiles, and models a dataset's distributions and multivariate relations."""
    
    def __init__(self):
        self.column_types = {}       # Name -> 'categorical' | 'numerical' | 'datetime'
        self.column_metadata = {}    # Detailed marginal stats and parameters
        self.copula_columns = []     # List of column names in the Gaussian Copula
        self.copula_covariance = None # Copula covariance matrix (Sigma)
        self.original_columns = []   # Order of original columns
        self.original_dtypes = {}    # Original pandas dtypes
        
    def fit(self, df: pd.DataFrame, random_seed=42):
        """Builds the statistical profile of the dataset."""
        np.random.seed(random_seed)
        self.original_columns = list(df.columns)
        self.original_dtypes = {col: str(df[col].dtype) for col in df.columns}
        
        # 1. Type identification & Marginal Profiling
        for col in df.columns:
            series = df[col]
            null_rate = series.isna().mean()
            
            # Detect type
            if detect_datetime_column(series):
                col_type = 'datetime'
            elif pd.api.types.is_bool_dtype(series):
                col_type = 'categorical'
            elif pd.api.types.is_numeric_dtype(series):
                col_type = 'numerical'
            else:
                col_type = 'categorical'
                
            self.column_types[col] = col_type
            
            # Handle marginals
            metadata = {
                'type': col_type,
                'null_rate': float(null_rate),
                'constant_value': None,
                'is_constant': False
            }
            
            non_null = series.dropna()
            
            if len(non_null) == 0:
                # Completely null column
                metadata['is_constant'] = True
                metadata['constant_value'] = None
                self.column_metadata[col] = metadata
                continue
                
            if len(non_null.unique()) == 1:
                # Constant column
                metadata['is_constant'] = True
                val = non_null.iloc[0]
                if col_type == 'datetime':
                    metadata['constant_value'] = str(pd.to_datetime(val))
                elif isinstance(val, (np.integer, int)):
                    metadata['constant_value'] = int(val)
                elif isinstance(val, (np.floating, float)):
                    metadata['constant_value'] = float(val)
                else:
                    metadata['constant_value'] = str(val)
                self.column_metadata[col] = metadata
                continue
                
            if col_type == 'categorical':
                # Frequency distribution (include nan as a class)
                if isinstance(series.dtype, pd.CategoricalDtype):
                    if '__NULL__' not in series.cat.categories:
                        series = series.cat.add_categories('__NULL__')
                counts = series.fillna('__NULL__').value_counts(normalize=True)
                
                # Separate __NULL__ to prepend it, keeping non-null categories in frequency order
                cats_list = list(counts.index)
                probs_list = [float(p) for p in counts.values]
                if '__NULL__' in cats_list:
                    null_idx = cats_list.index('__NULL__')
                    null_prob = probs_list[null_idx]
                    cats_list.pop(null_idx)
                    probs_list.pop(null_idx)
                    cats = ['__NULL__'] + cats_list
                    probabilities = [null_prob] + probs_list
                else:
                    cats = cats_list
                    probabilities = probs_list
                    
                metadata['categories'] = cats
                metadata['probabilities'] = probabilities
                
            elif col_type == 'numerical':
                metadata['min'] = float(non_null.min())
                metadata['max'] = float(non_null.max())
                metadata['mean'] = float(non_null.mean())
                metadata['std'] = float(non_null.std())
                # Store 1000 quantiles for empirical CDF reconstruction
                quantiles = np.percentile(non_null, np.linspace(0, 100, 1000))
                metadata['quantiles'] = [float(q) for q in quantiles]
                
            elif col_type == 'datetime':
                # Convert to unix timestamps
                ts = pd.to_datetime(non_null).astype('int64') // 10**9
                metadata['min'] = float(ts.min())
                metadata['max'] = float(ts.max())
                metadata['mean'] = float(ts.mean())
                metadata['std'] = float(ts.std())
                quantiles = np.percentile(ts, np.linspace(0, 100, 1000))
                metadata['quantiles'] = [float(q) for q in quantiles]
                
            self.column_metadata[col] = metadata
            
        # 2. Build Copula Latent Variables (Z)
        copula_dfs = []
        self.copula_columns = []
        
        for col in df.columns:
            meta = self.column_metadata[col]
            if meta['is_constant']:
                continue  # Skip constant columns in the joint copula
                
            series = df[col]
            col_type = self.column_types[col]
            
            if col_type == 'categorical':
                # Conditional mean mapping for categorical variables to prevent correlation attenuation
                cats = meta['categories']
                probs = meta['probabilities']
                cum_probs = np.cumsum([0.0] + probs)
                
                cat_to_z = {}
                for idx, cat in enumerate(cats):
                    lower_p = cum_probs[idx]
                    upper_p = cum_probs[idx+1]
                    
                    lp = max(1e-9, min(1.0 - 1e-9, lower_p))
                    up = max(1e-9, min(1.0 - 1e-9, upper_p))
                    a = scipy.stats.norm.ppf(lp)
                    b = scipy.stats.norm.ppf(up)
                    pdf_a = scipy.stats.norm.pdf(a) if lp > 1e-9 else 0.0
                    pdf_b = scipy.stats.norm.pdf(b) if up < 1.0 - 1e-9 else 0.0
                    p = up - lp
                    val = (pdf_a - pdf_b) / p if p > 0 else 0.0
                    cat_to_z[cat] = val
                    
                if isinstance(series.dtype, pd.CategoricalDtype):
                    if '__NULL__' not in series.cat.categories:
                        series = series.cat.add_categories('__NULL__')
                series_filled = series.fillna('__NULL__')
                z = series_filled.map(cat_to_z).astype(float).values
                
                # Standardize to unit variance
                z_mean = np.mean(z)
                z_std = np.std(z)
                if z_std > 0:
                    z = (z - z_mean) / z_std
                else:
                    z = np.zeros(len(series))
                    
                copula_dfs.append(pd.Series(z, name=col))
                self.copula_columns.append(col)
                
            elif col_type in ['numerical', 'datetime']:
                # For continuous: rank mapping for non-nulls
                non_null_mask = series.notna()
                non_null_vals = series[non_null_mask]
                
                if col_type == 'datetime':
                    non_null_vals = pd.to_datetime(non_null_vals).astype('int64') // 10**9
                    
                # Rank mapping
                ranks = non_null_vals.rank(method='average')
                u_vals = (ranks - 0.5) / len(ranks)
                u_vals = np.clip(u_vals, 1e-9, 1.0 - 1e-9)
                z_vals = scipy.stats.norm.ppf(u_vals)
                
                # Construct final z array (NaNs remain NaNs)
                z = np.full(len(series), np.nan)
                z[non_null_mask] = z_vals
                copula_dfs.append(pd.Series(z, name=col))
                self.copula_columns.append(col)
                
                # If there are missing values, add binary nullity indicator to Copula
                if meta['null_rate'] > 0:
                    nullity_col = f"{col}__is_null"
                    nullity_series = series.isna().astype(int)
                    # Nullity is binary: treat as categorical
                    n_counts = nullity_series.value_counts(normalize=True)
                    n_cats = list(n_counts.index)
                    n_probs = [float(p) for p in n_counts.values]
                    n_cum_probs = np.cumsum([0.0] + n_probs)
                    
                    n_to_z = {}
                    for idx, cat in enumerate(n_cats):
                        lower_p = n_cum_probs[idx]
                        upper_p = n_cum_probs[idx+1]
                        
                        lp = max(1e-9, min(1.0 - 1e-9, lower_p))
                        up = max(1e-9, min(1.0 - 1e-9, upper_p))
                        a = scipy.stats.norm.ppf(lp)
                        b = scipy.stats.norm.ppf(up)
                        pdf_a = scipy.stats.norm.pdf(a) if lp > 1e-9 else 0.0
                        pdf_b = scipy.stats.norm.pdf(b) if up < 1.0 - 1e-9 else 0.0
                        p = up - lp
                        val = (pdf_a - pdf_b) / p if p > 0 else 0.0
                        n_to_z[cat] = val
                        
                    z_nullity = nullity_series.map(n_to_z).astype(float).values
                    zn_mean = np.mean(z_nullity)
                    zn_std = np.std(z_nullity)
                    if zn_std > 0:
                        z_nullity = (z_nullity - zn_mean) / zn_std
                    else:
                        z_nullity = np.zeros(len(series))
                        
                    copula_dfs.append(pd.Series(z_nullity, name=nullity_col))
                    self.copula_columns.append(nullity_col)
                    
                    # Store nullity categories details in metadata
                    meta['nullity_categories'] = n_cats
                    meta['nullity_probabilities'] = n_probs
                    
        # 3. Compute Latent Covariance (Sigma)
        if copula_dfs:
            Z_df = pd.concat(copula_dfs, axis=1)
            # Pairwise pearson correlation
            corr = Z_df.corr(method='pearson')
            # Fill missing entries with 0.0 (e.g. correlation with constant parts or degenerate cases)
            corr = corr.fillna(0.0)
            
            # Get a writable copy of the numpy array to prevent read-only errors in modern Pandas
            corr_arr = corr.to_numpy().copy()
            np.fill_diagonal(corr_arr, 1.0)
            
            # Project to nearest Positive Semi-Definite matrix
            self.copula_covariance = make_psd(corr_arr)
        else:
            self.copula_covariance = np.array([[]])

    def to_dict(self):
        """Serializes the profiler's fitted state to a dict."""
        return {
            'column_types': self.column_types,
            'column_metadata': self.column_metadata,
            'copula_columns': self.copula_columns,
            'copula_covariance': self.copula_covariance.tolist() if self.copula_covariance is not None else [],
            'original_columns': self.original_columns,
            'original_dtypes': self.original_dtypes
        }
        
    def to_json(self, filepath):
        """Saves the profiler's state to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    def from_dict(self, d):
        """Loads state from a dictionary."""
        self.column_types = d['column_types']
        self.column_metadata = d['column_metadata']
        self.copula_columns = d['copula_columns']
        self.copula_covariance = np.array(d['copula_covariance']) if d['copula_covariance'] else None
        self.original_columns = d['original_columns']
        self.original_dtypes = d['original_dtypes']
        
    def from_json(self, filepath):
        """Loads state from a JSON file."""
        with open(filepath, 'r') as f:
            self.from_dict(json.load(f))
            
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generates a text report containing distribution tables and relationship matrices."""
        report = []
        report.append("# DATA DISTRIBUTION & RELATIONSHIP REPORT")
        report.append(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        
        # 1. Column Types
        report.append("## 1. Column Types")
        type_df = pd.DataFrame(list(self.column_types.items()), columns=["Column", "Type"])
        report.append(to_markdown_table(type_df) + "\n")
        
        # 2. Categorical Marginals
        report.append("## 2. Categorical Column Distributions")
        cat_cols = [c for c, t in self.column_types.items() if t == 'categorical']
        if not cat_cols:
            report.append("No categorical columns detected.\n")
        else:
            for col in cat_cols:
                meta = self.column_metadata[col]
                report.append(f"### Column: `{col}` (Null rate: {meta['null_rate']:.2%})")
                if meta['is_constant']:
                    report.append(f"- Constant value: `{meta['constant_value']}`\n")
                else:
                    cats = meta['categories']
                    probs = meta['probabilities']
                    # Take top 10 categories
                    cat_df = pd.DataFrame({"Category": cats, "Probability": probs})
                    cat_df['Percentage'] = cat_df['Probability'].apply(lambda x: f"{x:.2%}")
                    
                    if len(cat_df) > 10:
                        top_10 = cat_df.iloc[:10].copy()
                        other_prob = cat_df.iloc[10:]['Probability'].sum()
                        top_10 = pd.concat([top_10, pd.DataFrame([{"Category": "Other (Combined)", "Probability": other_prob, "Percentage": f"{other_prob:.2%}"}])], ignore_index=True)
                        report.append(to_markdown_table(top_10) + "\n")
                    else:
                        report.append(to_markdown_table(cat_df) + "\n")
                        
        # 3. Numerical & Datetime Marginals
        report.append("## 3. Numerical & Datetime Column Distributions")
        num_date_cols = [c for c, t in self.column_types.items() if t in ['numerical', 'datetime']]
        if not num_date_cols:
            report.append("No numerical or datetime columns detected.\n")
        else:
            summary_rows = []
            for col in num_date_cols:
                meta = self.column_metadata[col]
                if meta['is_constant']:
                    summary_rows.append({
                        "Column": col,
                        "Type": meta['type'],
                        "Null %": f"{meta['null_rate']:.2%}",
                        "Mean": "Constant",
                        "Std": "Constant",
                        "Min": str(meta['constant_value']),
                        "Max": str(meta['constant_value'])
                    })
                else:
                    if meta['type'] == 'datetime':
                        summary_rows.append({
                            "Column": col,
                            "Type": "datetime",
                            "Null %": f"{meta['null_rate']:.2%}",
                            "Mean": str(pd.to_datetime(meta['mean'], unit='s')),
                            "Std": f"{meta['std']:.2f} seconds",
                            "Min": str(pd.to_datetime(meta['min'], unit='s')),
                            "Max": str(pd.to_datetime(meta['max'], unit='s'))
                        })
                    else:
                        summary_rows.append({
                            "Column": col,
                            "Type": "numerical",
                            "Null %": f"{meta['null_rate']:.2%}",
                            "Mean": f"{meta['mean']:.4f}",
                            "Std": f"{meta['std']:.4f}",
                            "Min": f"{meta['min']:.4f}",
                            "Max": f"{meta['max']:.4f}"
                        })
            report.append(to_markdown_table(pd.DataFrame(summary_rows)) + "\n")
            
        # 4. Correlations & Associations
        report.append("## 4. Relationship Matrices")
        
        # Pearson (Linear - Numerical Only)
        num_cols = [c for c, t in self.column_types.items() if t == 'numerical']
        if len(num_cols) > 1:
            report.append("### Linear Correlation (Pearson) - Numerical Columns Only")
            pearson_corr = df[num_cols].corr(method='pearson')
            report.append(to_markdown_table(pearson_corr) + "\n")
            
        # Spearman (Rank / Monotonic - All Columns)
        report.append("### Rank Correlation (Spearman) - All Columns")
        # Temporary label encode for Spearman
        spearman_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            if self.column_types[col] == 'categorical':
                spearman_df[col] = df[col].astype('category').cat.codes.replace(-1, np.nan)
            elif self.column_types[col] == 'datetime':
                spearman_df[col] = pd.to_datetime(df[col]).astype('int64') // 10**9
            else:
                spearman_df[col] = df[col]
        spearman_corr = spearman_df.corr(method='spearman')
        report.append(to_markdown_table(spearman_corr) + "\n")
        
        # Cramér's V (Categorical Association)
        all_cat_cols = [c for c, t in self.column_types.items() if t == 'categorical']
        if len(all_cat_cols) > 1:
            report.append("### Categorical Association (Cramér's V)")
            v_matrix = pd.DataFrame(index=all_cat_cols, columns=all_cat_cols)
            for c1 in all_cat_cols:
                for c2 in all_cat_cols:
                    if c1 == c2:
                        v_matrix.loc[c1, c2] = 1.0
                    else:
                        v_matrix.loc[c1, c2] = cramers_v(df[c1], df[c2])
            report.append(to_markdown_table(v_matrix.astype(float)) + "\n")
            
        # Nullity Correlation (Missingness Patterns)
        null_cols = [c for c in df.columns if df[c].isna().sum() > 0]
        if len(null_cols) > 1:
            report.append("### Nullity Correlation (Pearson on missingness masks)")
            null_mask_df = df[null_cols].isna().astype(float)
            null_corr = null_mask_df.corr(method='pearson')
            report.append(to_markdown_table(null_corr) + "\n")
        else:
            report.append("### Nullity Correlation\nNo multiple columns with missing values found to analyze nullity correlation.\n")
            
        # Mutual Information (Non-linear general dependencies)
        report.append("### Non-Linear Association (Normalized Mutual Information)")
        nmi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
        for c1 in df.columns:
            for c2 in df.columns:
                if c1 == c2:
                    nmi_matrix.loc[c1, c2] = 1.0
                else:
                    nmi_matrix.loc[c1, c2] = binned_nmi(df[c1], df[c2])
        report.append(to_markdown_table(nmi_matrix.astype(float)) + "\n")
        
        return "\n".join(report)

    def save_report_and_matrices(self, df: pd.DataFrame, output_dir: str):
        """Saves the markdown report and raw matrices to the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Text report
        report_str = self.generate_report(df)
        with open(os.path.join(output_dir, "profile_report.md"), "w", encoding="utf-8") as f:
            f.write(report_str)
            
        # 2. Raw matrices as CSV
        # Pearson
        num_cols = [c for c, t in self.column_types.items() if t == 'numerical']
        if len(num_cols) > 1:
            df[num_cols].corr(method='pearson').to_csv(os.path.join(output_dir, "pearson_correlation.csv"))
            
        # Spearman
        spearman_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            if self.column_types[col] == 'categorical':
                spearman_df[col] = df[col].astype('category').cat.codes.replace(-1, np.nan)
            elif self.column_types[col] == 'datetime':
                spearman_df[col] = pd.to_datetime(df[col]).astype('int64') // 10**9
            else:
                spearman_df[col] = df[col]
        spearman_df.corr(method='spearman').to_csv(os.path.join(output_dir, "spearman_correlation.csv"))
        
        # Cramers V
        all_cat_cols = [c for c, t in self.column_types.items() if t == 'categorical']
        if len(all_cat_cols) > 1:
            v_matrix = pd.DataFrame(index=all_cat_cols, columns=all_cat_cols)
            for c1 in all_cat_cols:
                for c2 in all_cat_cols:
                    if c1 == c2:
                        v_matrix.loc[c1, c2] = 1.0
                    else:
                        v_matrix.loc[c1, c2] = cramers_v(df[c1], df[c2])
            v_matrix.astype(float).to_csv(os.path.join(output_dir, "cramers_v.csv"))
            
        # Nullity
        null_cols = [c for c in df.columns if df[c].isna().sum() > 0]
        if len(null_cols) > 1:
            df[null_cols].isna().astype(float).corr(method='pearson').to_csv(os.path.join(output_dir, "nullity_correlation.csv"))
            
        # NMI
        nmi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
        for c1 in df.columns:
            for c2 in df.columns:
                if c1 == c2:
                    nmi_matrix.loc[c1, c2] = 1.0
                else:
                    nmi_matrix.loc[c1, c2] = binned_nmi(df[c1], df[c2])
        nmi_matrix.astype(float).to_csv(os.path.join(output_dir, "normalized_mutual_information.csv"))


class DataReconstructor:
    """Generates synthetic datasets mirroring the statistical properties in a profile metadata dict."""
    
    def __init__(self):
        self.column_types = {}
        self.column_metadata = {}
        self.copula_columns = []
        self.copula_covariance = None
        self.original_columns = []
        self.original_dtypes = {}
        
    def load_metadata(self, metadata_path_or_dict):
        """Loads metadata dictionary or path to JSON."""
        if isinstance(metadata_path_or_dict, str):
            with open(metadata_path_or_dict, 'r') as f:
                d = json.load(f)
        else:
            d = metadata_path_or_dict
            
        self.column_types = d['column_types']
        self.column_metadata = d['column_metadata']
        self.copula_columns = d['copula_columns']
        self.copula_covariance = np.array(d['copula_covariance']) if d['copula_covariance'] else None
        self.original_columns = d['original_columns']
        self.original_dtypes = d['original_dtypes']
        
    def generate(self, num_rows: int, seed=42) -> pd.DataFrame:
        """Generates a synthetic DataFrame matching the saved profile statistics."""
        np.random.seed(seed)
        
        # 1. Handle joint sampling using Gaussian Copula
        copula_samples = {}
        if self.copula_covariance is not None and len(self.copula_columns) > 0:
            # Sample from multivariate normal
            means = np.zeros(len(self.copula_columns))
            Z = np.random.multivariate_normal(means, self.copula_covariance, size=num_rows)
            # Map to Uniform [0, 1] variables
            U = scipy.stats.norm.cdf(Z)
            
            for idx, col_name in enumerate(self.copula_columns):
                copula_samples[col_name] = U[:, idx]
                
        # 2. Reconstruct each column
        reconstructed_cols = {}
        
        for col in self.original_columns:
            meta = self.column_metadata[col]
            col_type = self.column_types[col]
            
            if meta['is_constant']:
                # Reproduce constant column
                val = meta['constant_value']
                if col_type == 'datetime' and val is not None:
                    reconstructed_cols[col] = pd.to_datetime([val] * num_rows)
                else:
                    reconstructed_cols[col] = [val] * num_rows
                continue
                
            # If the column is in the copula, retrieve its uniform values
            u = copula_samples.get(col)
            
            if col_type == 'categorical':
                if u is None:
                    # Fallback to univariate marginal random choice
                    cats = meta['categories']
                    probs = meta['probabilities']
                    cats_gen = np.random.choice(cats, size=num_rows, p=probs)
                else:
                    # Invert CDF for categorical using uniform variable
                    cats = meta['categories']
                    probs = meta['probabilities']
                    cum_probs = np.cumsum([0.0] + probs)
                    
                    cats_gen = []
                    # Search category for each sampled uniform value
                    # vectorized category assignment
                    idx_mapping = np.digitize(u, cum_probs) - 1
                    # Clip index just in case of rounding errors at boundary
                    idx_mapping = np.clip(idx_mapping, 0, len(cats) - 1)
                    cats_gen = [cats[idx] for idx in idx_mapping]
                    
                # Replace placeholder null markers with actual NaN
                cats_gen = [np.nan if val == '__NULL__' else val for val in cats_gen]
                reconstructed_cols[col] = cats_gen
                
            elif col_type in ['numerical', 'datetime']:
                # Reconstruct continuous variables
                if u is None:
                    # Fallback to standard uniform
                    u = np.random.uniform(0, 1, size=num_rows)
                    
                # Invert marginal using empirical quantiles
                quantiles = np.array(meta['quantiles'])
                # Interpolate from [0, 1] uniform to actual value space
                val_space = np.linspace(0, 1, len(quantiles))
                vals_gen = np.interp(u, val_space, quantiles)
                
                # Check for nullity helper in Copula
                nullity_col = f"{col}__is_null"
                if nullity_col in copula_samples:
                    # Invert binary nullity indicator from Copula
                    n_u = copula_samples[nullity_col]
                    n_cats = meta['nullity_categories']
                    n_probs = meta['nullity_probabilities']
                    n_cum_probs = np.cumsum([0.0] + n_probs)
                    
                    n_idx = np.digitize(n_u, n_cum_probs) - 1
                    n_idx = np.clip(n_idx, 0, len(n_cats) - 1)
                    is_null_gen = np.array([n_cats[idx] for idx in n_idx])
                    
                    # Apply missing mask (1 = null)
                    final_vals = np.where(is_null_gen == 1, np.nan, vals_gen)
                else:
                    # If not in copula, use raw null rate to assign missingness randomly
                    if meta['null_rate'] > 0:
                        null_mask = np.random.rand(num_rows) < meta['null_rate']
                        final_vals = np.where(null_mask, np.nan, vals_gen)
                    else:
                        final_vals = vals_gen
                        
                if col_type == 'datetime':
                    # Convert float timestamps back to datetime
                    datetime_series = pd.to_datetime(final_vals, unit='s', errors='coerce')
                    reconstructed_cols[col] = datetime_series
                else:
                    reconstructed_cols[col] = final_vals
                    
        # 3. Create DataFrame and enforce types
        recon_df = pd.DataFrame(reconstructed_cols, columns=self.original_columns)
        
        # Cast columns to original dtypes if possible
        for col in self.original_columns:
            dtype_str = self.original_dtypes[col]
            try:
                if 'int' in dtype_str:
                    # For integers with NaNs, pandas requires float or Int64 (nullable integer)
                    if recon_df[col].isna().sum() > 0:
                        recon_df[col] = recon_df[col].round().astype('Int64')
                    else:
                        recon_df[col] = recon_df[col].round().astype(dtype_str)
                elif 'float' in dtype_str:
                    recon_df[col] = recon_df[col].astype(dtype_str)
                elif 'bool' in dtype_str:
                    # Round and cast to boolean, using nullable boolean if NaNs are present
                    if recon_df[col].isna().sum() > 0:
                        recon_df[col] = recon_df[col].round().map({0.0: False, 1.0: True}).astype('boolean')
                    else:
                        recon_df[col] = recon_df[col].round().astype(bool)
                elif 'category' in dtype_str:
                    recon_df[col] = recon_df[col].astype('category')
                elif 'datetime' in dtype_str:
                    recon_df[col] = pd.to_datetime(recon_df[col])
                else:
                    recon_df[col] = recon_df[col].astype(dtype_str)
            except Exception as e:
                # Fallback to string if casting fails, to preserve data
                pass
                
        return recon_df
