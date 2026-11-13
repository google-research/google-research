# ==============================================================================
# generate_all_figures.py
# ------------------------------------------------------------------------------
# Batch Figure Rendering Orchestrator
# "Artificial Intelligence in High-Skill Knowledge Work: Evidence from Patent
#  Drafting and Prosecution"
# ------------------------------------------------------------------------------
# Role & Architecture:
#   Discovers and executes all 9 manuscript figure rendering modules (fig_*.py)
#   in sequential order. Each module reads serialized statistical outputs from
#   Jsons/ and renders publication-quality visualization plots to New/*.png.
#
# Usage:
#   python3 generate_all_figures.py
# ==============================================================================

import glob
import importlib.util
import os
import sys

def run_all_figs():
    """
    Discovers all fig_*.py scripts in the workspace, executes their render()
    functions, and reports status and error summaries.
    """
    fig_files = sorted(glob.glob("fig_*.py"))
    fig_files = [f for f in fig_files if os.path.isfile(f)]
    
    print(f"Found {len(fig_files)} figure generator scripts.")
    
    success = 0
    errors = []
    
    for f in fig_files:
        module_name = f[:-3]
        print(f"\n--- Running {f} ---")
        spec = importlib.util.spec_from_file_location(module_name, f)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'render'):
            raise AttributeError(f"No render() function found in {f}")
        module.render()
        print(f"Successfully generated figure from {f}")
        success += 1
            
    print("\n=======================================================")
    print(f"Finished generating figures: {success}/{len(fig_files)} succeeded.")
    print("=======================================================")

if __name__ == '__main__':
    run_all_figs()
