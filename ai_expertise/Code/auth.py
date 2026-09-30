# ==============================================================================
# auth.py
# ------------------------------------------------------------------------------
# Authentication Module for Remote Data Access and GitHub API Services
# "Artificial Intelligence in High-Skill Knowledge Work: Evidence from Patent
#  Drafting and Prosecution"
# ------------------------------------------------------------------------------
# Role & Architecture:
#   Handles credentials and authorization for Google Drive/Sheets and GitHub API
#   services across local and cloud environments (Google Colab).
# ==============================================================================

import os

try:
    from google.colab import auth
    from google.colab import userdata
    from google.auth import default
    import gspread
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def authenticate_google():
    """
    Authenticates with Google Drive and Sheets APIs if running inside Google Colab.
    Returns authorized gspread client instance, or None if in local execution mode.
    """
    if IN_COLAB:
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        return gc
    else:
        print("⚠️ Not running in Colab. Skipping Google authentication.")
        return None

def get_github_pat():
    """
    Retrieves GitHub Personal Access Token from Colab secrets or local .env file.
    Returns:
        str: Personal Access Token if set, or None.
    """
    if IN_COLAB:
        return userdata.get('GITHUB_PAT')
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get('GITHUB_PAT')
