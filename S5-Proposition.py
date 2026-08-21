import subprocess
import os
import sys
import logging
import json
import requests
import shutil
from cryptography.fernet import Fernet
from argparse import ArgumentParser

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class User:
    def __init__(self, name, role):
        self.name = name
        self.role = role

class AngelNETAutomation:
    def __init__(self, user, config_file):
        self.user = user
        self.env_path = f"/home/{user.name}/.angelnet_env"
        self.config_file = config_file
        self.config = self.load_config()
        self.repos = self.config.get('repositories', {})
        self.cipher = Fernet(self.load_encryption_key())

    def load_config(self):
        """Load configuration from a JSON file or use default settings."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as file:
                return json.load(file)
        logging.warning("Configuration file not found. Using default settings.")
        return {}

    def load_encryption_key(self):
        """Load encryption key from environment variable."""
        key = os.getenv('ENCRYPTION_KEY')
        if not key:
            raise ValueError("Encryption key not found.")
        return key.encode()

    def encrypt_data(self, data):
        """Encrypt data."""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data):
        """Decrypt data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    def setup_environment(self):
        """Create virtual environment if it doesn't exist."""
        if not os.path.exists(self.env_path):
            logging.info(f"Creating virtual environment for {self.user.name}...")
            subprocess.run([sys.executable, "-m", "venv", self.env_path], check=True)
        else:
            logging.info(f"Virtual environment already exists for {self.user.name}.")

    def install_dependencies(self, requirements_file):
        """Install dependencies from a requirements file."""
        logging.info(f"Installing dependencies for {self.user.name}...")
        subprocess.run([f"{self.env_path}/bin/pip", "install", "-r", requirements_file], check=True)

    def run_script(self, script_path):
        """Run a specified script and handle errors."""
        try:
            logging.info(f"Running script {script_path} for {self.user.name}...")
            result = subprocess.run([f"{self.env_path}/bin/python", script_path], check=True, capture_output=True, text=True)
            logging.info(f"Script output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Script failed with error: {e.stderr}")

    def apply_benefits(self):
        """Apply benefits based on the user's role and additional logic."""
        benefits = {
            "priority_support": False,
            "custom_solutions": False,
            "data_analytics": False,
            "exclusive_features": False,
            "extended_storage_bandwidth": False,
            "collaboration_tools": False,
            "enhanced_security": False
        }

        if self.user.role in ["Admin", "Institutional"]:
            benefits.update({
                "priority_support": True,
                "custom_solutions": True,
                "data_analytics": True,
                "exclusive_features": True,
                "extended_storage_bandwidth": True,
                "collaboration_tools": True,
                "enhanced_security": True
            })

            # Apply Quantitative Easing and Dark Side logic
            quantitative_easing_thing.handle_new_badge(self.user)
            dark_side_thing.enhance_security_features(self.user)
            dark_side_thing.adjust_storage_bandwidth(self.user)

            logging.info(f"Activating benefits for {self.user.name}:")
            for benefit, active in benefits.items():
                logging.info(f"- {benefit.replace('_', ' ').title()}: {'Enabled' if active else 'Disabled'}")

    def check_for_updates(self, repo_url):
        """Check for new updates in a repository."""
        try:
            response = requests.get(f"{repo_url}/releases/latest")
            response.raise_for_status()
            latest_version = response.json().get('tag_name', 'unknown')
            logging.info(f"Latest version for {repo_url} is {latest_version}.")
            return latest_version
        except requests.RequestException as e:
            logging.error(f"Failed to check for updates: {e}")
            return None

    def update_repository(self, repo_name):
        """Update a specified repository."""
        repo_url = self.repos.get(repo_name)
        if not repo_url:
            logging.error(f"Repository URL for {repo_name} not found.")
            return

        latest_version = self.check_for_updates(repo_url)
        if latest_version:
            logging.info(f"Updating {repo_name} to version {latest_version}...")
            try:
                zip_url = f"{repo_url}/archive/refs/tags/{latest_version}.zip"
                zip_path = f"/home/{self.user.name}/{repo_name}.zip"
                subprocess.run(["wget", zip_url, "-O", zip_path], check=True)
                shutil.unpack_archive(zip_path, f"/home/{self.user.name}/{repo_name}")
                os.remove(zip_path)
                logging.info(f"{repo_name} updated successfully.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to update {repo_name}: {e}")

    def automate_tasks(self):
        """Automate environment setup, dependency installation, benefit application, and script execution."""
        self.setup_environment()
        self.install_dependencies('requirements.txt')
        self.apply_benefits()
        self.run_script('quantitative_easing_thing.py')
        self.run_script('dark_side_thing.py')
        # Update repositories
        for repo_name in self.repos.keys():
            self.update_repository(repo_name)

def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="AngelNET Automation Script")
    parser.add_argument('--name', required=True, help="User's name")
    parser.add_argument('--role', required=True, choices=['Admin', 'Institutional', 'Researcher'], help="User's role")
    parser.add_argument('--config', default='config.json', help="Path to the configuration file")
    return parser.parse_args()

# Example usage
if __name__ == "__main__":
    args = parse_arguments()
    user = User(name=args.name, role=args.role)
    automation = AngelNETAutomation(user, args.config)
    automation.automate_tasks()
