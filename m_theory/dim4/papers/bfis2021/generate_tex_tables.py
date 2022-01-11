# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for generating large and detailed LaTeX tables describing equilibria.

Usage :
  m_theory/ $ python3 -i -m dim4.papers.bfis2021.generate_tex_tables

"""

import os
# For interactive debugging only.
import pdb  # pylint:disable=unused-import

from dim4.so8.src import dyonic
from m_theory_lib import m_util as mu
import numpy


_DEMO_TEX_OUTER = r'''\documentclass{article}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{array,longtable}
\newcolumntype{L}{>{$}l<{$}}
\newcolumntype{R}{>{$}r<{$}}
\newcolumntype{C}{>{$}c<{$}}
\begin{document}

\small\begin{longtable}{|L|R|L|C|C|C|C|}
\hline
\text{Name}&\text{Potential}&|\nabla V/g^2|&\text{Symmetry}&
\text{SUSY}&\text{BF-Stability}&\text{Discovered}\\
\hline
\input{so8c_summary.tex}
\end{longtable}

%%%%%%%%

\input{so8c_detailed.tex}
\end{document}
'''


def generate_tex_omega_pi8():
  """Produces LaTeX descriptions of solutions (omega=pi/8)."""
  sugra = dyonic.SO8c_SUGRA()
  numdata = mu.csv_numdata('dim4/so8/equilibria/SO8C_PI8_SOLUTIONS_GV.csv',
                           numdata_start_column=1)[1:]
  rows = [row[-70:] for row in numdata]
  with open('/tmp/so8c_equilibria.tex', 'wt') as h_out:
    h_out.write(_DEMO_TEX_OUTER)
  sugra.tex_all_solutions(
      rows,
      '/tmp/so8c',
      tag_kwargs=(('tag', r'S[{\scriptstyle 1/8}]'),
                  ('digits', 8),
                  ('scale', 1e5)),
      t_omega=mu.tff64(numpy.pi / 8))
  os.system(
      '(cd /tmp; pdflatex so8c_equilibria.tex; xdg-open so8c_equilibria.pdf)')


def generate_tex_omega_0():
  """Produces LaTeX descriptions of solutions (omega=0)."""
  sugra = dyonic.SO8c_SUGRA()
  numdata = mu.csv_numdata('dim4/so8/equilibria/SO8_SOLUTIONS.csv',
                           numdata_start_column=1)
  rows = [row[-70:] for row in numdata]
  with open('/tmp/so8c_w0_equilibria.tex', 'wt') as h_out:
    h_out.write(_DEMO_TEX_OUTER)
  sugra.tex_all_solutions(
      rows,
      '/tmp/so8c_w0')
  os.system(
      '(cd /tmp; pdflatex so8c_w0_equilibria.tex; '
      'xdg-open so8c_w0_equilibria.pdf)')


if __name__ == '__main__':
  generate_tex_omega_pi8()
  generate_tex_omega_0()
