# ==============================================================================
# macros.py
# ------------------------------------------------------------------------------
# Master Macro Compilation & LaTeX Variable Binding Engine
# "Artificial Intelligence in High-Skill Knowledge Work: Evidence from Patent
#  Drafting and Prosecution"
# ------------------------------------------------------------------------------
# Role & Architecture:
#   1. Executes the complete analysis pipeline via `analysis.run_all()`.
#   2. Consolidates point estimates, robust standard errors, p-values, sample sizes,
#      and hypothesis test statistics across all JSON payloads in Jsons/.
#   3. Generates the standard \csname macro bindings in New/macros.tex for direct
#      inclusion in latest.tex.
#   4. Verifies that every single macro reference `\stat{...}` in latest.tex is
#      properly populated with zero missing macro keys.
#
# Inputs:
#   - Jsons/*.json (all analytical payloads produced by analysis.py)
#   - latest.tex   (the primary manuscript file)
#
# Outputs:
#   - New/macros.tex (the compiled LaTeX macro definition file)
#   - macros.tex     (root copy)
# ==============================================================================

import json
import re
import glob
import os
import scipy.stats as stats

from analysis import run_all
from utils import push_to_github
from config import GITHUB_PAT

def fmt(v):
    """Formats numeric values to 2 decimal places as strings."""
    if v is None: return ''
    if isinstance(v, (int, float)): return f'{v:.2f}'
    return str(v)

def build_macros():
    """
    Executes the analytical engine, aggregates statistical outputs from all JSONs,
    reconstructs regression table macros, and standardizes key formatting.
    """
    print("Running analytical pipeline to gather statistical macros...")
    pipeline_macros = run_all()
    new_macros = dict(pipeline_macros)
    
    # Load explicitly saved JSONs to capture all statistical outputs
    for f in glob.glob('Jsons/*.json'):
        if 'models_data' in f or 'fig_' in f: continue
        with open(f, 'r') as jf:
            d = json.load(jf)
            if 'macros' in d: 
                new_macros.update(d['macros'])
        
    # Reconstruct regression table macros from Jsons/models_data.json
    with open('Jsons/models_data.json', 'r') as f:
        mod_data = json.load(f)
        
    for m_name, results in mod_data.items():
        if m_name == 'macros': continue
        models = results.get('models', [])
        for i, m in enumerate(models):
            if m is None: continue
            idx = f'm{i}'
            for var, val in m.get('params', {}).items():
                if var.startswith('C('): continue
                new_macros[f'{m_name}_coef_{var}_{idx}'] = fmt(val)
            for var, val in m.get('bse', {}).items():
                if var.startswith('C('): continue
                new_macros[f'{m_name}_SE_{var}_{idx}'] = fmt(val)
            for var, val in m.get('pvalues', {}).items():
                if var.startswith('C('): continue
                new_macros[f'{m_name}_pval_{var}_{idx}'] = fmt(val)
            for f_key, f_val in m.get('f_tests', {}).items():
                h_map = {
                    'junior = senior': 'F1',
                    'treat_x_junior = treat_x_senior': 'F2',
                    'junior + treat_x_junior = senior': 'F3',
                    'junior + treat_x_junior = senior + treat_x_senior': 'F4'
                }
                mapped_key = h_map.get(f_key, f_key)
                if 'pvalue' in f_val:
                    tval = f_val['tvalue']
                    df = m.get('df_resid', 100)
                    is_reversed = (
                        m_name in ['TimeOnTask', 'TimeOnTask_noFFE']
                        or (m_name in ['Patent', 'Patent_noFFE'] and i == 6)
                        or (m_name == 'Speed' and i in [2, 6, 10])
                    )
                    if is_reversed:
                        if mapped_key in ['F1', 'F2', 'F3', 'F4']:
                            pval = stats.t.cdf(tval, df)
                        else:
                            pval = f_val['pvalue']
                    else:
                        if mapped_key in ['F1', 'F2', 'F3', 'F4']:
                            pval = stats.t.sf(tval, df)
                        else:
                            pval = f_val['pvalue']
                    new_macros[f'{m_name}_{mapped_key}_{idx}'] = fmt(pval)
            if 'nobs' in m:
                new_macros[f'{m_name}_N_{idx}'] = str(int(m['nobs']))
                new_macros[f'{m_name}_N_group_binary_{idx}'] = str(int(m['nobs']))
                    
    # Generate the sample size macros for secondary surveys and administrative tables
    for tbl in ['DraftingSurvey', 'Patent', 'TimeOnTask', 'DraftingSurvey_noFFE', 'Patent_noFFE', 'TimeOnTask_noFFE']:
        if tbl in mod_data:
            max_obs = 0
            for m in mod_data[tbl].get('models', []):
                if m is not None and 'nobs' in m:
                    max_obs = max(max_obs, m['nobs'])
            if max_obs > 0:
                new_macros[f'{tbl}_max_n'] = f"{int(max_obs):,}"

    # Generate variance regressions and correlation macros from Jsons/additional_analysis.json
    with open('Jsons/additional_analysis.json', 'r') as f:
        add_data = json.load(f)
        
    if 'varregs' in add_data:
        v_map = {'Full Sample': 'ful', 'Juniors': 'jun', 'Seniors': 'sen'}
        o_map = {'tt1_sum_rat_drades': '10dayD', 'tt2_sum_rat_drades': '90dayD', 'tt2_sum_rat_cri': '90dayR'}
        for sg_name, sg_data in add_data['varregs'].items():
            if sg_name not in v_map: continue
            sg_short = v_map[sg_name]
            for out_name, out_data in sg_data.items():
                if out_name not in o_map: continue
                o_short = o_map[out_name]
                
                if 'Levene' in out_data:
                    stat = fmt(out_data['Levene'].get('stat'))
                    pval = fmt(out_data['Levene'].get('pval'))
                    new_macros[f"varregs_{o_short}_{sg_short}_lev_stat"] = stat
                    new_macros[f"varregs_{o_short}_{sg_short}_lev_pval"] = pval
                    
                    v_sg = 'all' if sg_name == 'Full Sample' else ('jun' if sg_name == 'Juniors' else 'sen')
                    v_out = {'tt1_sum_rat_drades': 'tt1_sum_drades', 'tt2_sum_rat_drades': 'tt2_sum_drades', 'tt2_sum_rat_cri': 'tt2_sum_cri'}[out_name]
                    new_macros[f"violin_levene_stat_{v_out}_{v_sg}"] = stat
                    new_macros[f"violin_levene_p_{v_out}_{v_sg}"] = pval
                    
                for q in ['q1', 'q2', 'q3', 'q4', 'q5']:
                    if q in out_data:
                        new_macros[f"varregs_{o_short}_{sg_short}_{q}_coef"] = fmt(out_data[q].get('coef'))
                        new_macros[f"varregs_{o_short}_{sg_short}_{q}_pval"] = fmt(out_data[q].get('pval'))

    for r_type in ['human', 'llm', 'pooled']:
        if r_type in add_data:
            for t_idx, t_data in enumerate(add_data[r_type]):
                t_title = t_data.get('m_key', t_data['title'].replace('-', '_').replace(' ', '_').lower())
                prefix = f"correlations_{r_type}_" if r_type != 'human' else "correlations_"
                new_macros[f"{prefix}alpha_{t_title}"] = fmt(t_data.get('alpha'))
                new_macros[f"{prefix}N_{t_title}"] = str(t_data.get('n_obs', ''))
                
                c_mat = t_data.get('c_mat', [])
                pairwise_corrs = []
                for i in range(len(c_mat)):
                    for j in range(i):
                        val = c_mat[i][j]
                        pairwise_corrs.append(val)
                        new_macros[f"{prefix}corr_{t_title}_{i+1}_{j+1}"] = fmt(val)
                if pairwise_corrs:
                    new_macros[f"{prefix}corr_{t_title}_min"] = fmt(min(pairwise_corrs))
                    new_macros[f"{prefix}corr_{t_title}_max"] = fmt(max(pairwise_corrs))
        
    # Alias fisher_all to fisher_full for latest.tex consistency
    for k in list(new_macros.keys()):
        if k.startswith("fisher_") and "_all" in k:
            new_macros[k.replace("_all", "_full")] = new_macros[k]
            
    # Final pass: Standardize numeric values to 2 decimal places
    for k, v in list(new_macros.items()):
        v_str = str(v)
        if re.search(r'^-?\d+\.\d{3,}$', v_str):
            try:
                new_macros[k] = f"{float(v):.2f}"
            except (ValueError, TypeError):
                pass
            
    return new_macros

def main():
    """
    Coordinates compilation of macros, outputs New/macros.tex, and verifies
    coverage against manuscript citations.
    """
    new_macros = build_macros()
    
    # Sort macros alphabetically and build the LaTeX string
    latex_lines = []
    for k in sorted(new_macros.keys()):
        val = new_macros[k]
        latex_lines.append(f"\\expandafter\\def\\csname {k}\\endcsname{{{val}}}")
        
    latex = "\n".join(latex_lines) + "\n"
    
    # Save locally
    with open('macros.tex', 'w', encoding='utf-8') as f:
        f.write(latex)
    if os.path.exists('New'):
        with open('New/macros.tex', 'w', encoding='utf-8') as f:
            f.write(latex)
        
    if os.path.exists('draft_2026Sep23.tex') and os.path.getsize('draft_2026Sep23.tex') > 0:
        tex_file = 'draft_2026Sep23.tex'
    elif os.path.exists('draft_2026Sep1.tex'):
        tex_file = 'draft_2026Sep1.tex'
    else:
        tex_file = 'draft_2026Aug20.tex' if os.path.exists('draft_2026Aug20.tex') else ('latest.tex' if os.path.exists('latest.tex') else 'v16.tex')
    print(f"\nChecking macros used in {tex_file} against generated macros...")
    missing_in_tex = set()
    if os.path.exists(tex_file):
        with open(tex_file, 'r', encoding='utf-8') as f:
            tex_content = f.read()
            # Strip comments before finding macros
            tex_content = re.sub(r'(?m)%.*$', '', tex_content)
            used_macros = set(re.findall(r'\\stat\{([^}]+)\}', tex_content))
            for m in used_macros:
                if m not in new_macros:
                    missing_in_tex.add(m)
            
        if missing_in_tex:
            print(f"\n[WARNING] {len(missing_in_tex)} macros referenced in {tex_file} are missing from the generated macros:")
            for m in sorted(list(missing_in_tex)):
                print(f"  - {m}")
        else:
            print(f"\n[INFO] All macros referenced in {tex_file} are present and verified.")
    
    print("\nPushing New/macros.tex to GitHub...")
    import config, auth
    github_pat = auth.get_github_pat()
    c_dict = {
        'OWNER': config.GITHUB_CONFIG['OWNER'], 
        'REPO': config.GITHUB_CONFIG['REPO'], 
        'TARGET_DIR': config.GITHUB_CONFIG['TARGET_DIR'],
        'BRANCH': config.GITHUB_CONFIG.get('BRANCH', 'main')
    }
    push_to_github("macros.tex", latex, github_pat=github_pat, config=c_dict)
    print("Macro compilation complete!")

if __name__ == '__main__':
    main()
