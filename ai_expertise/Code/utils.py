# ==============================================================================
# utils.py
# ------------------------------------------------------------------------------
# General Utility Functions & API Helpers for InFlow Replication Pipeline
# "Artificial Intelligence in High-Skill Knowledge Work: Evidence from Patent
#  Drafting and Prosecution"
# ------------------------------------------------------------------------------
# Role & Architecture:
#   1. Provides statistical formatting helpers (`fv` for floats, `stars` for significance).
#   2. Implements directional hypothesis F-test / t-test evaluation (`get_f_test_results`).
#   3. Manages local filesystem serialization and optional GitHub API synchronization
#      (`push_to_github`, `upload_plot`).
#   4. Provides robust dataset loading from local data/ with GitHub fallback
#      (`fetch_csv_with_fallback`).
# ==============================================================================

import requests
import base64
import io
import os
import glob
import urllib.parse
import time
import hashlib
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import scipy.stats as stats

# Configured resilient session for GitHub API synchronization
git_session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[403, 409, 422, 429, 500, 502, 503, 504]
)
git_session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

def compute_git_blob_sha(content_str):
    """
    Computes Git blob SHA-1 hash for git object comparison.
    """
    content_bytes = content_str.encode('utf-8')
    header = f"blob {len(content_bytes)}\0".encode('utf-8')
    return hashlib.sha1(header + content_bytes).hexdigest()

def push_to_github(filename, content, github_pat, config):
    """
    Saves text content (e.g. LaTeX tables or macros) to local target directory
    and optionally commits and pushes to GitHub repository if PAT is available.

    Parameters:
        filename (str): Name of the output file (e.g., 'firm_summary.tex').
        content (str): Text content to write.
        github_pat (str, optional): GitHub Personal Access Token.
        config (dict): Target GitHub repository configuration.
    """
    os.makedirs(config['TARGET_DIR'], exist_ok=True)
    local_path = os.path.join(config['TARGET_DIR'], filename)
    with open(local_path, "w", encoding="utf-8") as lf:
        lf.write(content)
    print(f"  [LOCAL] ✅ Saved {filename} to local {config['TARGET_DIR']} directory.")

    if not github_pat: 
        return
    
    time.sleep(1.5)
    
    safe_name = urllib.parse.quote(filename)
    api_url = f"https://api.github.com/repos/{config['OWNER']}/{config['REPO']}/contents/{config['TARGET_DIR']}/{safe_name}"
    
    headers = {
        'Authorization': f'token {github_pat}', 
        'Accept': 'application/vnd.github.v3+json',
        'Cache-Control': 'no-cache'
    }
    
    sha = None
    r_get = git_session.get(api_url, headers=headers, timeout=10)
    if r_get.status_code == 200: 
        file_data = r_get.json()
        sha = file_data.get('sha')
        decoded_content = base64.b64decode(file_data.get('content', '')).decode('utf-8')
        if decoded_content == content:
            print(f"  [GIT STATUS] ⚠️ Content unchanged for {filename}. Skipping remote push.")
            return
    elif r_get.status_code not in (200, 404):
        raise RuntimeError(f"GitHub GET SHA check failed for {filename}: HTTP {r_get.status_code} - {r_get.text}")
        
    payload = {
        "message": f"Update {filename}", 
        "content": base64.b64encode(content.encode('utf-8')).decode('utf-8'), 
        "branch": config['BRANCH']
    }
    if sha: 
        payload["sha"] = sha
    
    res = git_session.put(api_url, headers=headers, json=payload, timeout=15)
    if res.status_code in [200, 201]: 
        commit_sha = res.json().get('commit', {}).get('sha', 'Unknown')
        print(f"  ✅ Pushed {filename} to {config['TARGET_DIR']}. Commit SHA: {commit_sha}")
    else: 
        raise RuntimeError(f"  ❌ Push failed for {filename}: HTTP {res.status_code} - {res.text}")

def fetch_csv_with_fallback(file_prefix, github_pat, config):
    """
    Loads CSV dataset from the local data/ directory if available; otherwise falls back
    to fetching directly from the GitHub repository API.

    Parameters:
        file_prefix (str): Prefix of the target CSV file (e.g. 'maindata').
        github_pat (str, optional): GitHub Personal Access Token.
        config (dict): Repository configuration dictionary.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if os.path.exists('data'):
        local_files = sorted(glob.glob(f"data/{file_prefix}*.csv"), reverse=True)
        if local_files:
            print(f"  [LOCAL] Loading {file_prefix} from {local_files[0]}")
            return pd.read_csv(local_files[0])
            
    api_dir_url = f"https://api.github.com/repos/{config['OWNER']}/{config['REPO']}/contents/data?ref={config['BRANCH']}"
    auth_headers = {'Authorization': f'token {github_pat}', 'Accept': 'application/vnd.github.v3+json'} if github_pat else {}
    dl_headers = auth_headers.copy()
    if dl_headers:
        dl_headers['Accept'] = 'application/vnd.github.v3.raw'
        
    resp = requests.get(api_dir_url, headers=auth_headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch CSV directory list for {file_prefix}: HTTP {resp.status_code}")
    files = resp.json()
    if not isinstance(files, list):
        raise RuntimeError(f"Unexpected response format for CSV directory list: {files}")
    matching = sorted([f['name'] for f in files if f['name'].startswith(file_prefix) and f['name'].endswith('.csv')], reverse=True)
    if not matching:
        return pd.DataFrame()
    raw_url = f"https://raw.githubusercontent.com/{config['OWNER']}/{config['REPO']}/{config['BRANCH']}/data/{matching[0]}"
    r = requests.get(raw_url, headers=dl_headers)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download raw CSV for {file_prefix}: HTTP {r.status_code}")
    return pd.read_csv(io.StringIO(r.text))

def upload_plot(plt_obj, filename, github_pat, config):
    """
    Saves a matplotlib plot figure to the local target directory and optionally
    uploads to GitHub repository.

    Parameters:
        plt_obj (matplotlib.pyplot): Matplotlib pyplot module or active figure.
        filename (str): Name of the image file (e.g., 'fisher.png').
        github_pat (str, optional): GitHub Personal Access Token.
        config (dict): Target GitHub repository configuration.
    """
    os.makedirs(config['TARGET_DIR'], exist_ok=True)
    local_path = os.path.join(config['TARGET_DIR'], filename)
    plt_obj.savefig(local_path, dpi=300, bbox_inches='tight')
    print(f"  [LOCAL] ✅ Saved {filename} to local {config['TARGET_DIR']} directory.")
    
    if not github_pat:
        return
        
    time.sleep(1.5)
    buf = io.BytesIO()
    plt_obj.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.read()
    
    safe_name = urllib.parse.quote(filename)
    api_url = f"https://api.github.com/repos/{config['OWNER']}/{config['REPO']}/contents/{config['TARGET_DIR']}/{safe_name}"
    headers = {
        'Authorization': f'token {github_pat}',
        'Accept': 'application/vnd.github.v3+json',
        'Cache-Control': 'no-cache'
    }
    
    sha = None
    r_get = git_session.get(api_url, headers=headers, timeout=10)
    if r_get.status_code == 200:
        sha = r_get.json().get('sha')
    elif r_get.status_code not in (200, 404):
        raise RuntimeError(f"GitHub GET check failed for {filename}: HTTP {r_get.status_code} - {r_get.text}")
        
    payload = {
        "message": f"Update {filename}",
        "content": base64.b64encode(img_bytes).decode('utf-8'),
        "branch": config['BRANCH']
    }
    if sha:
        payload["sha"] = sha
        
    res = git_session.put(api_url, headers=headers, json=payload, timeout=20)
    if res.status_code in [200, 201]:
        commit_sha = res.json().get('commit', {}).get('sha', 'Unknown')
        print(f"  ✅ Uploaded {filename} to {config['TARGET_DIR']}. Commit SHA: {commit_sha}")
    else:
        raise RuntimeError(f"  ❌ Upload failed for {filename}: HTTP {res.status_code} - {res.text}")

def get_var_label(df_cb, var_name):
    """
    Retrieves the descriptive label for a variable from the codebook.
    """
    if df_cb is None or df_cb.empty or 'varname' not in df_cb.columns:
        return var_name
    match = df_cb[df_cb['varname'] == var_name]
    if not match.empty:
        if 'label_shorthand' in df_cb.columns and pd.notna(match['label_shorthand'].values[0]):
            return match['label_shorthand'].values[0]
        if 'label' in df_cb.columns and pd.notna(match['label'].values[0]):
            return match['label'].values[0]
    return var_name

def fv(v):
    """
    Formats a numeric float/integer value to 2 decimal places.
    """
    if v is None or pd.isna(v) or v == "": 
        return ""
    try:
        return f"{float(v):.2f}"
    except (ValueError, TypeError):
        return str(v)

def stars(p):
    """
    Computes standard academic significance stars based on p-value:
      - p < 0.01: ***
      - p < 0.05: **
      - p < 0.10: *
    """
    if p is None or pd.isna(p) or p == "": 
        return ""
    try:
        p = float(p)
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
    except (ValueError, TypeError):
        pass
    return ""

def get_f_test_results(res, test_string, h0_inequality):
    """
    Evaluates one-sided hypothesis test p-values from regression results.

    Parameters:
        res: Fitted regression results object.
        test_string (str): Linear restriction (e.g., 'treat_x_junior = treat_x_senior').
        h0_inequality (str): String indicating direction of alternative ('>' vs '<').

    Returns:
        tuple: (f_value, one_sided_p_value)
    """
    try:
        t_res = res.t_test(test_string)
        t_val = t_res.tvalue.item() if hasattr(t_res.tvalue, 'item') else float(t_res.tvalue)
        df_res = res.df_resid
        
        if "<" in h0_inequality:
            p_val = stats.t.cdf(t_val, df_res)
        elif ">" in h0_inequality:
            p_val = stats.t.sf(t_val, df_res)
        else:
            p_val = t_res.pvalue.item() if hasattr(t_res.pvalue, 'item') else float(t_res.pvalue)
            
        f_val = t_val ** 2
        return f_val, p_val
    except Exception as e:
        return None, None

def load_exclusion_list(file_path):
    """
    Reads a list of exclusion identifiers (practitioner UUIDs, firm names, or firm numbers)
    from a text file (one item per line). Ignores blank lines and comments starting with '#'.

    Parameters:
        file_path (str): Path to exclusion text file.

    Returns:
        set: Cleaned lowercase set of exclusion entries.
    """
    if not os.path.exists(file_path):
        print(f"  [EXCLUSIONS] ⚠️ Exclusion file '{file_path}' not found. No exclusions loaded from this file.")
        return set()
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().lower() for line in f]
    items = {line for line in lines if line and not line.startswith('#')}
    print(f"  [EXCLUSIONS] Loaded {len(items)} exclusion items from '{file_path}'.")
    return items

def get_firm_exclusion_mappings(df, excluded_terms, unique_id_var='email', firm_var='firm_num'):
    """
    Resolves firm exclusion terms (anonymized names or numbers) into corresponding
    anonymized firm names, firm numbers, and practitioner UUIDs found in the dataset.

    Parameters:
        df (pd.DataFrame): Primary dataset (e.g. df_main or df_dates_raw).
        excluded_terms (set): Set of lowercase firm names or firm numbers.
        unique_id_var (str): Unique practitioner ID column name (default 'email').
        firm_var (str): Firm identifier column name (default 'firm_num').

    Returns:
        tuple: (matched_names, matched_nums, matched_emails, matched_domains)
    """
    if not excluded_terms:
        return set(), set(), set(), set()

    matched_names = set(excluded_terms)
    matched_nums = set(excluded_terms)
    matched_domains = set(excluded_terms)
    matched_emails = set()

    if df is None or df.empty:
        return matched_names, matched_nums, matched_emails, matched_domains

    # Scan dataframe columns for direct matches
    fn_col = next((c for c in df.columns if 'firm' in c.lower() and 'name' in c.lower()), 'firm_name')
    fnum_col = next((c for c in df.columns if 'firm' in c.lower() and ('num' in c.lower() or 'number' in c.lower())), firm_var)
    email_col = next((c for c in df.columns if 'email' in c.lower()), unique_id_var)

    for _, row in df.iterrows():
        f_name = str(row.get(fn_col, '')).strip().lower() if fn_col in df.columns else ''
        f_num = str(row.get(fnum_col, '')).strip().lower().replace('.0', '') if fnum_col in df.columns else ''
        email = str(row.get(email_col, '')).strip().lower() if email_col in df.columns else ''

        if (f_name in excluded_terms or 
            f_num in excluded_terms or 
            f_name in matched_names or
            f_num in matched_nums):
            if f_name: matched_names.add(f_name)
            if f_num: matched_nums.add(f_num)
            if email: matched_emails.add(email)

    return matched_names, matched_nums, matched_emails, matched_domains


