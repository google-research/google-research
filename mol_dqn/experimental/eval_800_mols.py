# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Run evaluation on 800 molecules.

Run pretrained model on all 800 molecules and record the optimization path.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
from absl import app
from absl import flags
from absl import logging
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Contrib import SA_Score
from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
from mol_dqn.chemgraph.mcts import deep_q_networks
from mol_dqn.chemgraph.mcts import molecules as molecules_mdp


flags.DEFINE_float('sim_delta', 0.0, 'similarity_constraint')
flags.DEFINE_integer('num_episodes', 50, 'episodes.')
flags.DEFINE_float('gamma', 0.999, 'discount')
flags.DEFINE_string('model_folder', '800_mols_0830', 'model_folder')
FLAGS = flags.FLAGS

all_mols = [
    r'COc1cc2c(cc1OC)CC([NH3+])C2',
    r'C[C@@H]1CC[C@@H](C(N)=O)CN1C(=O)c1nnn[n-]1',
    r'CC[NH+]1CC[C@@H](CNCc2ccc([O-])c[nH+]2)C1', r'OC[C@@H](Br)C(F)(F)Br',
    r'CNC(=O)/C(C#N)=C(/[O-])C1=NN(c2cc(C)ccc2C)C(=O)CC1',
    r'C[NH+](C)CCS[C@@H]1C[C@H](C(C)(C)C)CC[C@@H]1C#N',
    r'CN(c1ncnc(N2CCN(c3cccc[nH+]3)CC2)c1[N+](=O)[O-])C1CC[NH+](C)CC1',
    r'COc1cc(C[NH+]2CC[C@@H]([NH+]3CCCC3)C2)ccc1OCC(=O)N1CCCC1',
    r'COCCN1C[C@@]23C=C[C@@H](O2)[C@H](C(=O)N(C)Cc2cnccn2)[C@H]3C1=O',
    r'COc1ccc(/C=C2\SC(=O)N(CC(=O)NCC(=O)[O-])C2=O)cc1OC',
    r'COCC[NH+]1CC[C@H]2CCCC[C@@H]2C1',
    r'CCC[NH+](C1CCC([NH3+])CC1)[C@H]1CCOC1',
    r'CC(C)CNC(=O)[C@H](C)[NH+]1CCCN(CC[NH3+])CC1',
    r'CN(CC[C@@H]1CCC[C@]1(N)C#N)CC[NH+]1CCCC1',
    r'OC[C@H]1C[NH+](Cc2ccccc2)CCC12OCCO2',
    r'CC[C@@H](O)[C@@]1(C[NH3+])CCC[C@H](C)C1', r'OCc1cn2c(n1)OC(Cl)=CC2',
    r'CCn1ccnc(N2CCCC[C@@H](N3CC[NH+](C)CC3)C2)c1=O',
    r'COCCOC[C@H]1CC[NH+](C2C[C@@H](C)O[C@H](C)C2)C1',
    r'NC(=O)C1(N2CCCC2)CC[NH2+]CC1',
    r'CC[C@H](C)[NH+]1[C@@H](C(=O)[O-])CC[C@H]2CCCC[C@H]21',
    r'C=CCn1c(C)nn(C[NH+]2CCC[C@H](C(=O)NCCC)C2)c1=S',
    r'O=C(N[C@@H](C(=O)[O-])c1ccccc1)C1CCC(CNC(=O)[C@@H]2Cc3ccccc3C[NH2+]2)CC1',
    r'COC[C@H](O)C[NH+]1CCC(C)(C)C1',
    r'C[C@@H](C(=O)[O-])[C@@H](N[S@](=O)C(C)(C)C)C(C)(C)C',
    r'O=c1n(CCO)c2ccccc2n1CCO',
    r'Cc1ccc(C[NH+](C)[C@@H](C)C(=O)NCCc2ccc3c(c2)OCCO3)o1',
    r'Cc1[nH+]cn(C[C@H](C)[C@H]2CC[NH+]3CCC[C@H]23)c1C',
    r'O=C([O-])c1ccc(CNC(=O)c2cnns2)o1', r'C=CCC[C@@H](C)[NH+](C)CCc1nccs1',
    r'C[C@H](Cn1ccnc1)NS(C)(=O)=O',
    r'CNC(=O)[C@@H]1CCCN(S(=O)(=O)c2c(C)nn(CC(=O)NC(C)(C)C)c2C)C1',
    r'CSCC(=O)NNC(=O)NC[C@@]1([NH+](C)C)CCC[C@H](C)C1',
    r'CCNC(=O)c1cccc(NC(=O)C[NH+](C)CC)c1',
    r'CCC[NH2+][C@@H]1COC[C@H]1C(=O)NCc1cscc1C',
    r'C=C(C)CN/C(N)=[NH+]\Cc1ccc(C)cc1N(C)C',
    r'CCO[C@@H]1C[C@@H]([NH+](C)C[C@@H]2CCCN(S(C)(=O)=O)C2)C12CCCCC2',
    r'N#Cc1cn(C[C@H]2CCCC[C@H]2O)c(=O)nc1[O-]',
    r'C[C@H](CSc1ccc(C(=O)N(C)C)cn1)C(=O)[O-]',
    r'CC[C@H]1CN(C(=O)[C@@H]2CC[C@@H]3CCCC[C@@H]3[NH2+]2)CCN1C',
    r'CCO[C@@H]1C[C@@H]([NH3+])[C@@H]1Nc1ncc(Cl)cc1F',
    r'Cc1cnn(CCCNC(=O)N2CCCC[C@H](N3CC[NH+](C)CC3)C2)c1',
    r'Cc1nn(C)c(CO[C@@H]2CCC[C@@H]([NH3+])C2)c1Cl',
    r'C[C@@H]1CC[C@@H](C(=O)[O-])[C@H]2C(=O)N(c3ccccc3)C(=O)[C@@H]21',
    r'Cc1ccc(NC(=O)C(=O)N2CC[C@H]([NH+]3CCCC3)C2)cc1C(=O)N(C)C',
    r'CNc1ccccc1C(=O)N1CCN2C(=O)NC[C@H]2C1',
    r'C[C@H]1[C@@H](C)SCC[NH+]1Cc1cccc2cn[nH]c12',
    r'CC(C)[C@H](O)[C@]1(C[NH3+])CCc2ccccc21',
    r'CCC[NH2+][C@]1(C(=O)OCC)CC[C@H](n2cc(Cl)c(C)n2)C1',
    r'Cc1cc([C@H]2CCC[NH+]2CC(=O)NC(N)=O)no1',
    r'COc1ccc(Cc2noc(C3CC[NH2+]CC3)n2)cn1',
    r'Cc1ccc(C[NH+]2CCC(N3CCC(C(=O)N4CCOCC4)CC3)CC2)o1',
    r'COc1ccccc1CC(=O)N1C[C@H]2CC[C@@H]1CN(S(C)(=O)=O)C2',
    r'CC1=C(C(=O)C2=C([O-])C(=O)N(CC[NH+](C)C)[C@H]2c2cccc(Cl)c2)[C@H](C)N=N1',
    r'CC(C)(C)OC(=O)N1CCc2cccc(C[NH+]3CC[C@H]([N+]4=CCCC4)C3)c21',
    r'O=C(NC[C@@H]1CCC[NH+](Cc2ccccc2F)C1)c1nc[nH]n1',
    r'Cc1ccc(CCN2C[C@]34C=C[C@H](O3)[C@H](C(=O)N3CC(O)C3)[C@H]4C2=O)cc1',
    r'COc1cccc(C(=O)NCC[NH+](C)C2CCCCC2)c1F',
    r'CC(C)[NH+]1CCN(CC(=O)NCCc2ccc(F)cc2)CC1',
    r'C[C@@H](CO)NC(=O)NC[C@H]1Cc2ccccc2O1',
    r'CC(C)(O)CC[NH2+][C@H]1CCCS(=O)(=O)C1',
    r'CC1(C)C(=O)NCC[NH+]1Cc1ccc(OCC(F)F)cc1',
    r'N#C[C@H]1CN(C(=O)[C@H]2CNCc3ccccc32)CCO1',
    r'O=C(NC[C@@H]1CCC[NH+](CC2=c3ccccc3=[NH+]C2)C1)[C@H]1NN=C2C=CC=C[C@H]21',
    r'COC(=O)c1cc(NC(=O)[C@H]2CC[NH2+][C@H]2C)ccc1F',
    r'C[C@H]1[C@H](C(=O)[O-])CCN1S(=O)(=O)[C@@H](C)C#N',
    r'O=C([O-])COc1ccccc1/C=N/NC(=O)C1CC1',
    r'CC(C)c1nc(C(=O)N2CCC[C@@H]([NH+]3CCCC3)C2)n[nH]1',
    r'Cn1cc[nH+]c1N1CCN(C[C@@H](O)c2cccc(Br)c2)CC1',
    r'Cc1nn(C)c(C)c1-c1cc(C(=O)N[C@@H]2CC[C@H]([NH+](C)C)C2)n[nH]1',
    r'C[C@H]1CN(C(=O)[C@@H]2CCS(=O)(=O)C2)C[C@H](C)O1',
    r'CCOC(=O)C1=C(N)N(C)c2ccccc2[C@@]12C(=O)OC(C)=C2C(C)=O',
    r'O=C(NC1CC1)[C@@H]1CCC[NH+](C2CCN(C(=O)c3ccc[nH]3)CC2)C1',
    r'CC[C@](C)(NC(=O)[C@](C)(N)c1ccccc1)C(=O)[O-]',
    r'CCc1noc(C)c1C[NH+](C[C@@H]1CCCCO1)C(C)C',
    r'CCC(CC)([C@H](Cc1nc(C)cs1)NC)[NH+]1CCCC1',
    r'CC(C)(C)CS(=O)(=O)N1Cc2nc[nH]c2C[C@H]1C(=O)[O-]',
    r'C[NH2+][C@@]1(C(=O)[O-])CC[C@H](Sc2nccc(=O)[nH]2)C1',
    r'COc1ncccc1C(=O)NC[C@H]1C[C@H](O)C[NH+]1Cc1ccccc1',
    r'CC(C)C[C@@H](C[NH3+])c1nc(C2CCOCC2)no1',
    r'O=C1N(C[NH+]2CCN(c3ccccc3)CC2)c2ccccc2C12O[C@@H]1CCCC[C@H]1O2',
    r'CCOC(=O)C1(C#N)CC(OC)(OC)C1', r'[NH3+][C@@H](CSCc1nccs1)C(=O)[O-]',
    r'CC(C)[C@@H]([NH2+][C@@H](C)CS(C)(=O)=O)c1cccnc1',
    r'CCCN[C@]1(C#N)CC[C@H](N2C[C@H](C)OC[C@@H]2C)C1',
    r'C[C@H](O)CC#CC[NH+]1CCC[C@H](c2cccnc2)C1',
    r'CC(C)OCCS(=O)(=O)N[C@@H]1CCCCC[C@H]1[NH3+]',
    r'CN1CCO[C@@H](CN(C)C2(C[NH3+])CCCCC2)C1',
    r'Cc1c(C[NH+]2CCC[C@H]2c2ccc3c(c2)OCO3)cc(C#N)n1C',
    r'CC[NH+]1CCC2(CC1)OC[C@H](C(=O)[O-])N2C(=O)c1ccc(F)cc1',
    r'CC[NH+](CCNC(=O)N[C@H]1CC(=O)N(C(C)(C)C)C1)C(C)C',
    r'CC[C@@H]([NH2+]CCN1CCCS1(=O)=O)c1ccc(OC)cc1',
    r'CCCn1ncc(C[NH2+]C)c1C(F)(F)F',
    r'C[C@H]1C[NH+]2CCCC[C@@H]2CN1C(=O)NC[C@@H](C)C(=O)[O-]',
    r'C[C@@H]1CC[C@@H](O)[C@H]([NH+](C)CCOCC2CC2)C1',
    r'[NH3+]C[C@H]1CCC[C@H]1S(=O)(=O)c1cccc(F)c1',
    r'C[C@H]1[NH2+]CCC[C@@H]1NC(=O)c1cccc(OC(F)F)c1',
    r'CC[C@H]1C[C@H](C)CC[C@@H]1[NH2+]CCCN1CCCC1=O',
    r'CCN[C@@H]1[C@H]([NH+]2CCC[C@H]3CCC[C@@H]32)CCC1(C)C',
    r'FC(F)n1ccnc1CN1CC[NH+](CCN2CCOCC2)CC1',
    r'O[C@@H]1C[C@@H](c2nc(C3CC3)no2)[NH+](Cc2c[nH]c3ccccc23)C1',
    r'Cc1ccc(-c2ccncc2)cc1NC(=O)C(=O)N[C@H]1CC[C@@H]([NH+](C)C)C1',
    r'CC1CCC(C[NH3+])(NC(=O)N[C@H]2CCOC2)CC1',
    r'O[C@H](C1CC[NH+](Cc2c(Cl)nc3ccccn23)CC1)C(F)(F)F',
    r'C[N+]1(/N=C(\[S-])NN)CCOCC1',
    r'NC(=O)CN1c2ccccc2C(=O)N[C@H]1c1cc(Cl)cc([N+](=O)[O-])c1[O-]',
    r'C[C@@H](NC(=O)c1cc(C[NH+]2CCC(O)CC2)on1)c1cn(C)c2ccccc12',
    r'Cc1cscc1C[NH2+][C@H](C)CS(C)(=O)=O',
    r'C[C@@H](C(=O)C1=c2ccccc2=[NH+]C1)[NH+]1CCC[C@@H]1[C@@H]1CC=CS1',
    r'CNS(=O)(=O)CC(=O)N[C@H]1CCCN(c2ccccc2)C1',
    r'C[S@@](=O)c1ccc(C[NH+]2CCC(OC[C@H]3CCCO3)CC2)cc1',
    r'CCN(CC)c1ccc(N)c(N)[nH+]1', r'C[NH2+]C[C@H]1C[C@H]1c1ccccc1Br',
    r'C=CC(=O)OCC(C)(C)C[NH+](C)C',
    r'COCc1cc([C@@H](C)NC2CC[NH+]([C@@H]3CCCC[C@@H]3O)CC2)ccc1OC',
    r'C[C@@H]1CN(Cc2noc(-c3ccc(F)cc3)n2)CC[C@@H]1[NH3+]',
    r'COc1ccc(OC)c([C@@H](O)Cc2[nH+]ccn2C)c1',
    r'Cc1nnc(S[C@H](C)C(=O)N2CCOCC2)n1C', r'[NH3+][C@H](Cc1ccc(O)cc1)c1ncccn1',
    r'CC[NH+]1CCN(C[C@@H](C)CNC(=O)NCc2sccc2C)CC1',
    r'CC[C@@H](O)[C@@H]1CCCC[NH+]1Cc1nc2ccccc2n1CC',
    r'C[C@H]1CN(S(=O)(=O)[C@@H](C)c2cnccn2)CC[NH2+]1',
    r'O=C([O-])C1([C@@H]2CCCC[C@H]2O)CCOCC1',
    r'C[NH2+][C@]1(C(=O)[O-])CCC[C@@H](OCC2CCCCC2)C1',
    r'CC[NH+](CCO[C@H]1CCCCO1)CC1CC[NH2+]CC1',
    r'CC(=O)N1CCc2cc(S(=O)(=O)N[C@@H](C(=O)[O-])C(C)C)ccc21',
    r'COc1ccc(-c2ccc(C[NH2+][C@@H]3CC[C@H]([NH+](C)C)C3)o2)c([N+](=O)[O-])c1',
    r'CC1(C)O[C@@H]2O[C@@H]3OC(C)(C)O[C@H]3[C@@H]2O1',
    r'COc1ccccc1[C@@H]1C[NH+](Cc2cc(C(C)=O)cn2C)C[C@H]1C(=O)[O-]',
    r'CCO[C@@H](C)c1noc(CN2CC[NH+]([C@H]3CCCc4ccccc43)CC2)n1',
    r'CCN1CCN(C(=O)[C@H]2[C@@H]3C[C@H]4[C@H](OC(=O)[C@H]42)[C@H]3Cl)CC1',
    r'CC(C)CNC(=O)[C@H](C)[NH+]1CCC[C@@H]1[C@@H]1CCCCC1=O',
    r'CC[C@@H](CSC)[NH+](C)Cn1nc(-c2cccs2)[nH]c1=S',
    r'CC(C)(C)c1ccc([C@@]2(C)C[NH+]=C(N)N2CC2CC2)cc1',
    r'Cc1nnc(CCC[NH+]2CCC(CC[NH+]3CCCC[C@@H]3C)CC2)o1',
    r'C[C@H]1C[C@H]([NH2+]Cc2ccccn2)CS1',
    r'C=C(C)[C@@](C)(O)C#CC[NH+]1CCCC[C@@H]1c1cccnc1',
    r'CC[C@](C)(C[NH3+])[C@H](O)c1ccc2c(c1)OCO2',
    r'C[C@H]1CCCN(C(=O)C2C(C)(C)C2(C)C)[C@@H]1C[NH3+]',
    r'O=[N+]([O-])c1ccc([C@@H]2OC[NH+]3COC[C@@H]23)cc1',
    r'COC(=O)[C@@H]1NS(=O)(=O)c2ccsc2C1=O',
    r'O=C(c1nnn[n-]1)N1CCC[C@@H]1[C@@H]1CCC[NH2+]1',
    r'O=C([O-])CC1=C(C(=O)[O-])CCCC1', r'Cn1cc(C(=O)NCCc2ccccc2)c(C(=O)[O-])n1',
    r'Cc1nn(C)c(C)c1CN[C@H]1CCC[NH2+]C1',
    r'CC(C)[C@H]1C[NH2+]CC[C@]12CCO[C@H](C)C2',
    r'COc1ccc(S(=O)(=O)N(CC(=O)N2CC[NH2+]CC2)C(C)C)cc1',
    r'CCC1(CO)CC[NH+](Cc2cc(OC)c(O)cc2Br)CC1',
    r'CCc1nc2n(n1)CCC[C@@H]2N[C@H]1CCN(C2CC2)C1=O',
    r'CC(C)C[C@H](C[NH+](C)C)Nc1ncncc1N',
    r'Oc1ccccc1/C=[NH+]/CCC/[NH+]=C/c1ccccc1O',
    r'CC(C)C[C@](C)(O)CNC(=O)C1CCC(C[NH3+])CC1',
    r'CC(=O)Nc1ccc(CN2CC[NH+](C3CCCC3)[C@H](CCO)C2)cc1',
    r'COc1cc(Cl)cc(CN2C[C@@H]3CCC[NH+]3C[C@H]2C)c1OC',
    r'CN1CCN(c2ncc(C[NH2+]C(C)(C)C)s2)C(C)(C)C1=O',
    r'CC1CCN(S(=O)(=O)c2ccc(C(=O)N3CCC[C@@H]3C(=O)[O-])cc2)CC1',
    r'Cc1cc(NC(=O)C(=O)NC[C@H]2CC[NH+](C)C2)ccc1OC(C)C',
    r'C#CC[NH2+]CC(=O)Nc1cccc(-c2nncn2C)c1',
    r'NC(=O)c1cccc(CNC(=O)[C@@H]2C[C@H]3CC[C@@H]2O3)c1',
    r'C/[NH+]=C(/NCc1cc(C)on1)NCc1ccccc1-n1ccnc1',
    r'COc1ccccc1CC(=O)N[C@@H]1CS(=O)(=O)C[C@H]1Cl',
    r'C[NH2+][C@@H](Cc1csc(C)n1)C(OC)OC', r'COc1c(F)cc([C@H]([NH3+])CO)cc1Cl',
    r'Cc1ccc(-c2nc3nc(CN4CC[NH+](C)CC4)cc([O-])n3n2)cc1',
    r'COc1ccc([N+](=O)[O-])cc1CN1CCC[NH+](CC(=O)N2CCCC2)CC1',
    r'C[C@@H](c1cccc(-c2ccc(C3(O)CC[NH2+]CC3)cc2)c1)[NH+]1CCCC1',
    r'CCC[C@@H](c1nnnn1C[C@@H]1CCCO1)[NH+]1CCN(c2cc(C)ccc2C)CC1',
    r'CC[NH+](CC)C[C@H]1CC[NH2+]C1', r'C[C@@H]([NH3+])C(=O)N1CC[C@@H](O)C1',
    r'C#CC(C)(C)NC[C@H]1CN(C)CCO1',
    r'CCOC1CC[NH+](CC[C@@H](O)c2ccc(C)c(F)c2)CC1',
    r'COC[C@@H](C)NC(=O)C[NH+]1CCc2sc(-c3csc(C)n3)cc2C1',
    r'CCOc1cccc(NC(=O)/C(C#N)=C/[C@H]2C=c3ccccc3=[NH+]2)c1',
    r'C[C@H](Cn1cccn1)[NH2+]CC(=O)N1C[C@H](C(N)=O)Oc2ccccc21',
    r'C[C@@H](C[NH+]1CCCCC1)NC(=O)c1n[nH]c(C2CC2)n1',
    r'C[C@H]1C[C@@H](C)C[NH+](C[C@@H](O)CO[C@@H]2CCC[C@H]2C)C1',
    r'Cc1csc([C@H](C)NC(=O)CCC[NH+]2CCCCC2)n1',
    r'O[C@]1(C[NH2+][C@@H]2CCN(CC(F)(F)F)C2)CCCc2ccccc21',
    r'CCNS(=O)(=O)[C@@H]1CC[NH+](C[C@@H]2CCCc3ccccc32)C1',
    r'Cc1nc(CCC(F)(F)F)[nH]c(=O)c1CCC(=O)[O-]',
    r'COc1ncnc2c1nc([C@@H](C)Cl)n2[C@H](C)C(N)=O',
    r'O=C([O-])[C@H]1C=C[C@H](NS(=O)(=O)c2ccc3c(c2)CCO3)C1',
    r'CCC[C@H]1CCC[NH+](CCCS)CC1',
    r'CC[S@](=O)[C@@H]1CCC[C@H](NC(=O)NNC(=O)C(C)(C)C)C1',
    r'COc1cc(C[NH+]2CCC[C@]3(CCC(=O)N(C4CC4)C3)C2)cc(OC)c1',
    r'Cc1ccccc1CC[NH+]1[C@H]2CC[C@@H]1CC(=O)C2',
    r'C[C@@H](NC1CC[NH+]([C@H]2CCCC[C@@H]2O)CC1)c1ccsc1',
    r'Cc1ccc([C@@](C)(O)CNC(=O)NC[C@@H](c2ccco2)[NH+]2CCCCC2)o1',
    r'C[C@@H]1[C@H](C(=O)[O-])CCN1S(=O)(=O)c1ccc(F)c(Cl)c1',
    r'COC(=O)/C=C/c1ccc[n+]([O-])c1',
    r'[NH3+]C[C@H]1CC[C@@H](C(=O)N(C[C@@H]2CCCO2)[C@H]2CCSC2)O1',
    r'N#Cc1nn(C(N)=O)c(N)c1C#N', r'C[C@H](CNC(=O)N1CCN(S(C)(=O)=O)CC1)N1CCOCC1',
    r'[NH3+]C[C@@H](c1ccc(F)cc1)[C@@H]1CCS(=O)(=O)C1',
    r'CC(C)CNC(=O)NC(=O)[C@H](C)[NH2+]CC1(N2CCSCC2)CCCC1',
    r'[NH3+][C@@H](CO)c1cc(C(F)(F)F)cc([N+](=O)[O-])c1[O-]',
    r'CCCCN1C(=O)[C@@H]2[C@H](CCC(=O)[O-])'
    r'N[C@]3(C(=O)Nc4c(CC)cccc43)[C@@H]2C1=O',
    r'Cc1cc(C)nc(NC(=[NH2+])Nc2ccc(S(=O)(=O)Nc3nccc(C)n3)cc2)n1',
    r'Cn1cnnc1C[NH+]1CC[C@]2(CCCN(C3CCCC3)C2=O)C1',
    r'CCN(CC(C)(C)O)C(=O)[C@H]1C[C@@H]2C=C[C@H]1C2',
    r'CC(C)C(=O)NCC[NH2+][C@@H](C)C[C@@H]1CCCCC[NH2+]1',
    r'CC(C)C[NH2+]C[C@H]1CCCO[C@H]1[C@H]1CC=CCC1',
    r'Cc1nc2c(s1)[C@H]([NH+](C)Cc1cccn1C)CCC2',
    r'COc1ccc([C@@H]2CN(C(=O)Cn3nc4ccccc4n3)C[C@H]2[NH+](C)C)cc1',
    r'CC(=O)NCCn1c(SCC(=O)[O-])nc2cccnc21',
    r'C[C@@H]([NH2+]C[C@@H](C)[S@](C)=O)c1ccc(F)c(F)c1',
    r'C[NH2+]C1CCN(C(=O)C2(C)CCCC2)CC1',
    r'CCC(CC)[C@H](CNC(=O)c1cnc2c(C)cccn2c1=O)[NH+](C)C',
    r'C[NH2+]C[C@H](C)[C@H](C)n1cccn1',
    r'CC(C)[NH+]1CCCN(C(=O)C(C)(C)c2cccc(C#N)c2)CC1',
    r'CC(C)CN1CCO[C@H](CNC(=O)N2CCN(C(=O)[C@H]3C[C@H]3C)CC2)C1',
    r'CNS(=O)(=O)[C@H]1CCC[NH+]([C@@H](C)c2ncc(C(C)(C)C)o2)C1',
    r'Cc1[nH+]c(NC[C@@H](C)C[C@H](C)O)ccc1I',
    r'COC(=O)C(C)(C)C[NH2+][C@@H]1C[C@H]1c1ccccc1',
    r'C[C@@H]1SCCC[C@]1(C[NH3+])N1CCC(C[NH+](C)C)CC1',
    r'CCC[NH2+][C@H](Cc1nn(C)c2ccccc12)c1ncc[nH]1',
    r'O=C([O-])[C@@H]1C[C@@H]1C(=O)N1CC2(CCCC2)c2c(F)cccc21',
    r'COc1cc(Br)c(C[NH2+]C[C@H](C)O)cc1OC',
    r'COc1ccc(N2/C(=N/C(=O)CCCC(=O)[O-])S[C@@H]3CS(=O)(=O)C[C@H]32)cc1Cl',
    r'CC[C@@H](C)[NH+]1CCN([C@H]2CC[C@H]([NH2+]C)C2)CC1',
    r'CCCN1CC(=O)N2[C@@H](CC3=c4ccccc4=[NH+][C@@H]3[C@H]2c2ccccc2OC)C1=O',
    r'Cc1noc(C[NH+]2CC[C@@H](OCCCc3ccccc3)C2)n1',
    r'CCc1nn(C)c(C[C@]2(C[NH3+])CCCC(C)(C)[C@@H]2O)c1Cl',
    r'COc1cc(NC(=O)C[C@@H]2C[NH2+]CCO2)cc(OC)c1',
    r'COC1CC[NH+](CCNc2nccn(C)c2=O)CC1',
    r'C[C@@H](O)c1cc(F)ccc1N(C)C[C@@H]1CCC[NH+]1C',
    r'OCCC#Cc1cc(F)cc(C[NH+]2CCC[C@H]2CO)c1',
    r'C[C@H]1[NH2+]CC[C@@H]1c1nncn1C1CCCCC1',
    r'CC[NH+]1CCN(Cc2nc3nc(C)cc(N4CCCCCC4)n3n2)CC1',
    r'COc1cc(C)ccc1NC(=O)N[C@H]1CC[C@H]([NH+](C)C)C1',
    r'O=C(NCCCc1n[nH]c(=O)[nH]1)[C@H]1C[C@@H]1c1c(F)cccc1F',
    r'CCN(CCOC)[C@]1(C[NH3+])CCCS[C@@H]1C',
    r'CC(C)c1cc(NC(=O)C[NH+]2CCC[C@H]2c2cccs2)on1',
    r'[NH3+][C@@H](Cc1cccc(F)c1F)[C@H]1CN2CCC[C@@H]2CO1',
    r'C[C@H](CCO)[NH2+][C@H]1CCc2c(Br)cccc21',
    r'CC(C)[C@@H](NC(=O)c1ccc(NS(C)(=O)=O)cc1)C(=O)[O-]',
    r'COc1ccccc1[C@H](C)NC(=O)C[NH+](C)C1CCS(=O)(=O)CC1',
    r'COCc1ccc(C[NH+](C)Cc2ccccc2O)o1',
    r'COC(=O)c1sccc1NC(=O)[C@@H]1CC[NH2+][C@@H]1C',
    r'COCCCS(=O)(=O)/N=C(\[O-])c1cnn(C(C)C)c1C',
    r'[N-]=[N+]=NC[C@H](Nc1ccc(Br)cc1)C(=O)[O-]',
    r'Cc1ccc([C@H](NC(=O)NC[C@H]2CC[NH+](C3CC3)C2)C2CC2)cc1',
    r'Cc1ccc(F)cc1NC(=O)C(=O)NCCCn1cc[nH+]c1',
    r'CC[NH+]1CCC[C@@H]1CNC(=O)N1CCC[C@H]([NH+](C)C)C1',
    r'Cc1nsc(N2CCC[NH+](Cc3ncccc3C)CC2)n1',
    r'Cc1nc(C[NH+]2CCC[C@@H]2CN2CCOCC2)cs1',
    r'Cc1c(C[NH+]2CC[C@@H](N3CCOCC3)[C@H](O)C2)[nH]c2c(C)cccc12',
    r'CCN[C@@H]1c2cc(OC)ccc2C[C@H]1[NH+](C)C(C)C',
    r'COc1cc(C[NH+]2CC[C@@H](NCc3scnc3C)C2)cc(OC)c1',
    r'CC[C@H](C)[C@H]1OCCC[C@H]1C(=O)[O-]',
    r'Cc1c(Cl)cccc1S(=O)(=O)NC[C@@H](C)CN1CC[NH+](C)CC1',
    r'CC(C)[C@@H]1C(=O)NCC[NH+]1Cc1cc(F)cc(F)c1',
    r'C[C@@H](C[C@@H]1CCCC[NH2+]1)[NH2+]C[C@@H]1CCCC[NH+]1C',
    r'[NH3+]C[C@](O)(CN1CC[NH+]2CCCC[C@H]2C1)C1CC1',
    r'Cc1ccc([N+](=O)[O-])cc1NC(=O)C(=O)N1CC[C@H]([NH+]2CCCC2)C1',
    r'CC(C)C[NH+](C)CCC(=O)[O-]', r'OCc1ccc(CN2CCCC[C@@H]([NH+]3CCCC3)C2)o1',
    r'CCCn1nnnc1CN1CC[C@]2(C1)NC(=O)N(C(C)C)C2=O',
    r'C[C@H]1CN(c2ccccc2C[NH2+]C2CCC(O)CC2)C[C@@H](C)O1',
    r'CC[C@H](C)C[C@H](C)NC(=O)[C@@H]1CCC[NH2+][C@@H]1C',
    r'COCC[N+]1=C(C)C[C@@H](C(=O)CSc2nccc([O-])n2)[C@H]1C',
    r'CC[NH2+][C@H]([C@@H]1CN2CCC[C@@H]2CO1)[C@@H]1CCCC[C@H]1CC',
    r'CCC(CC)([C@@H](Cc1ccc[nH+]c1N)NC)[NH+](C)C',
    r'CN(C)C(=O)O[C@@H]1CCCC[C@H]1C[NH3+]',
    r'Nc1ccc(Cl)c(S(=O)(=O)N2CCN3C(=O)NC[C@H]3C2)c1',
    r'COc1ccc(C[NH+]2CCNC(=O)CC2)cc1OCC(C)C',
    r'O=C(NCC[NH+]1CCN(C(=O)c2ccccc2[N+](=O)[O-])CC1)[C@H]1CC(=O)N(c2ccccc2)C1',
    r'[NH3+][C@H](C(=O)[O-])[C@H](O)c1ccc(F)cc1',
    r'CN(C)C(=O)C[C@H](C[NH3+])N1CCOCC1(C)C',
    r'C[C@H](CN(C)C(=O)c1ccc(F)c(F)c1F)C(=O)[O-]',
    r'CCc1ccc([C@@H](C)NC(=O)[C@H](C)[NH+]2CCc3n[nH]cc3C2)cc1',
    r'CC(=O)C1=C([O-])C(=O)N(CCC2=c3ccccc3=[NH+][C@H]2C)[C@H]1c1ccccc1F',
    r'[NH3+][C@H]1CCC[C@@H]1CCSc1n[nH]c(=O)n1C1CC1',
    r'CC(C)Cc1nc(SCC(=O)NC[C@@H]2CCCO2)c2c(=O)n(C)c(=O)n(C)c2n1',
    r'CC[NH+]1CCC[C@H](NC(=O)c2ccc(OC)c(O)c2)C1',
    r'C[NH+]1CC[C@@H](NC(=O)NCCS(C)(=O)=O)[C@@H]1c1ccc(Cl)cc1',
    r'CC(C)Cn1cc[nH+]c(N2C[C@H]3CC[C@@H]2C3)c1=O',
    r'CCOC(=O)C1CCC(NC(=O)[C@@](C)([NH3+])CC)CC1',
    r'C[C@H](CCCO)NC(=O)C[C@H]1CCS(=O)(=O)C1',
    r'O=C(c1cc(COc2ccc(-n3cncn3)cc2)on1)N1CC[NH+]2CCC[C@H]2C1',
    r'CC(C)[C@H]1CN(C(=O)CCC(N)=O)CCC[NH+]1Cc1ccc(F)cc1',
    r'N#Cc1ccnc(N2CCC([NH2+]C[C@@H]3CCCO3)CC2)c1', r'[NH3+]CC(=O)c1ccc(F)cc1',
    r'CNC(=O)[C@H](C)C[NH+](C)Cc1nnc2n(C)c(=O)c3cc(C)ccc3n12',
    r'CC[NH+]1CCC[C@@H]1C[NH+](C)CCC(=O)[O-]',
    r'CCNC(=O)CN(CC)C(=O)CC1([NH3+])CCC1',
    r'CC1=C(CC[NH+]2CCC[C@@H](C(N)=O)C2)C(C)(C)CCC1',
    r'COc1cccc(O[C@@H]2CC[C@H]([NH3+])C2)n1',
    r'C[S@](=O)CC[NH2+]C/C=C/c1ccc(C#N)cc1',
    r'CN[C@]1(C#N)CCC[C@H]([NH+](C)CCc2ccccn2)C1',
    r'Cc1nn(-c2ccccc2)c(C)c1CNC(=O)[C@@H]1[C@@H](C(=O)[O-])[C@@H]2C=C[C@H]1C2',
    r'CCC[C@H](C)NC(=O)C[NH2+]Cc1cscc1C',
    r'C/[NH+]=C(/NCc1[nH+]ccn1CC(C)C)N1C[C@H]2CC=CC[C@@H]2C1',
    r'CC(C)[NH2+]CC1CC[NH+](CCSc2ccccc2)CC1',
    r'C[C@@H](Cc1[nH+]ccn1C)C[C@@H](C)Br',
    r'C[C@H](SCC[NH3+])c1ccc(C(=O)[O-])o1',
    r'Cc1ccc2ncc(C(=O)N(C)C3CC[NH+](C(C)C)CC3)n2c1',
    r'COCCn1c2ccccc2n2c(=O)n(CC(=O)N(C)C)nc12',
    r'CC1(C)CCC[C@@H]1NC(=O)COCC(=O)[O-]',
    r'CC1=C[C@@H](C)[C@H]2C(=O)N([C@H](Cc3ccccc3)C(=O)[O-])C(=O)[C@H]2C1',
    r'COC(=O)C[C@H](NC(=O)N1CC[NH+]2CCC[C@@H]2C1)C(=O)[O-]',
    r'c1cnc2c(O[C@H]3CCC[NH2+]C3)cccc2c1',
    r'C[C@@H]1CCC[NH+](C[C@@H](C)NC(=O)N[C@@H]2CCCC[C@@H]2n2cccn2)C1',
    r'CCC[NH2+][C@H]1[C@H](S(=O)(=O)C(C)C)CCC1(C)C',
    r'CN(CC[NH+](C)C)C(=O)C[C@H]1COCCN1C(=O)c1ccc2[nH]nnc2c1',
    r'CC(C)CNC(=O)CNC(=O)[C@H]1CCC[NH+]1Cc1ccc(F)cc1',
    r'CC1CCN(C(=O)C[NH+]2CCC[C@@H](c3nc4ccccc4o3)C2)CC1',
    r'CCCCOc1ccccc1C[C@@H]([NH3+])C(=O)[O-]',
    r'NC(=O)c1n[nH]c2ccc(NC(=O)C(=O)NCC[C@H]3C[C@H]4CC[C@@H]3C4)cc12',
    r'O[C@H]1CCC2C1C1CC[C@H](O)C21', r'CCc1nn(C)c(C(=O)NC[C@@H](CC)CCO)c1N',
    r'CC(C)CN1C(=O)C2(CCCC2)[NH2+]C12CCN(C(=O)c1ccoc1)CC2',
    r'COc1ccc(NC(=O)[C@H]2C[C@H]3CC[C@@H]2O3)cn1',
    r'CCOC(=O)N[C@@H]1CCCN(C(=O)NC[C@H]2CC[NH+](C3CC3)C2)C1',
    r'CNc1nc(C2CCN(C(=O)Cc3ccccn3)CC2)[nH+]c2c1CN(C(C)=O)CC2',
    r'C[NH+](C)[C@@H]1CCC[C@@H](NC(=O)NCc2cc(=O)[nH]c3ccccc23)C1',
    r'C[NH+](C)[C@@H]1CC[C@@H](NC(=O)c2ccc3c(c2)CCCN3S(C)(=O)=O)C1',
    r'CC(C)[C@H](CO)NS(=O)(=O)c1ccsc1C(=O)[O-]',
    r'C[C@H]1CCC[C@H](NC(=O)CN2CC[NH+](C/C=C/c3ccco3)CC2)C1',
    r'NC(=O)CN1CC2(CCC1=O)CC[NH+](Cc1ccnc3ccccc13)CC2',
    r'C[NH+](CC(=O)N1CCOC1=O)CC1CC[NH2+]CC1',
    r'CCCn1c(C[C@@]2(O)CCC[NH2+]C2)nc2ccccc21',
    r'CC[NH2+][C@@]1(C(=O)OC)CCC[C@@H](Oc2ccccc2)C1',
    r'CCCC[C@@H](C(=O)N1CCN(CC(=O)N2CCCC[C@@H]2C)CC1)N1CCCS1(=O)=O',
    r'C[NH+](C)[C@@H]1CC[C@H](NC(=O)[C@@H]2CCCc3[nH]ncc32)C1',
    r'CC(C)CCNC(=O)[C@@H](C)Oc1ccc(N)cc1C(=O)[O-]',
    r'O=C(NCCCN1CCOCC1)c1nc(-c2cnccn2)no1',
    r'CCC[NH2+][C@H](Cc1ccccc1)[C@@H]1CN(CC)CCO1',
    r'Cn1nncc1C[NH+](CC1CCCCC1)C1CC1',
    r'C[C@H]1C[C@H]([NH+]2CC[C@H](S(=O)(=O)NC3CC3)C2)CC(C)(C)C1',
    r'NC(=O)[C@]1([NH2+]C2CC2)CC[C@H](Oc2cc(F)cc(F)c2)C1',
    r'O=C(NC[C@H]1CCS(=O)(=O)C1)N1CCC[C@H]([NH+]2CCCC2)C1',
    r'COc1ccc(OC)c([C@H]2CC[NH+](c3c([O-])c(=[OH+])c3=O)C2)c1',
    r'CC(C)C[C@@H]([NH3+])C(=O)N1CC[C@H](C(=O)[O-])[C@@H]1C',
    r'O=C(c1cccc(F)c1)N1CCCC[C@@H]1c1nc2c(c(=O)[nH]1)C[NH+](Cc1cccnc1)CC2',
    r'CC1CCC(C#N)([C@H](O)C=O)CC1',
    r'C=CCN(CC(=O)[O-])C(=O)[C@@H](C[NH3+])C(C)C',
    r'CS[C@H]1CC[C@H](NC(=O)N[C@@H](C)Cn2cc[nH+]c2)C1',
    r'CC(C)n1cc(S(=O)(=O)N2CCn3c(nn(C)c3=O)C2)cn1',
    r'C[NH+](Cc1ccsc1)Cc1ccc(C(N)=O)cc1[N+](=O)[O-]',
    r'CC(C)[C@H](C)[NH+]1Cc2cccc(NCc3cc(=O)n4ccsc4[nH+]3)c2C1',
    r'O=C([O-])CCNC(=O)N[C@@H]1CCOC1', r'O=C([O-])COCCNC(=O)[C@@H]1CCCCO1',
    r'CCCCn1nc(C)c(C[NH2+]C[C@@H](C)O)c1Cl',
    r'O=C(CCCn1cncn1)N1CCC[C@@H](N2CCNC2=O)C1',
    r'C[C@H]1CN(C(=O)NCc2ncnn2C)C[C@@H](C)O1',
    r'CCc1[nH+]n(C)c2c1NC[C@]1(C)COC[C@H]1N2',
    r'CCNC(=O)N1CC[C@@H]([NH2+][C@H](C)CC(=O)Nc2cccc(F)c2)C1',
    r'CC(C)[NH+]1CCC(N2CC[NH+](Cc3c(F)ccc(F)c3F)C[C@@H]2CCO)CC1',
    r'CC[NH+]1C[C@H](c2ccccc2)CC2(CCN(C(=O)c3ccon3)CC2)C1',
    r'[NH3+]C1([C@H]2CCO[C@@]3(CCSC3)C2)CCCC1',
    r'CC(C)([C@H](O)c1ccc(Cl)s1)[NH+]1CCCC1',
    r'CC(C)N[C@@]1(C#N)CC[C@@H](N(C)C[C@@H]2CC[NH+](C)C2)C1',
    r'CCC(=O)N1CCCC[C@@H]1C(=O)NC[C@@H](C1CC1)[NH+](C)C',
    r'Cc1nc2n(n1)C[C@H]([NH2+]C[C@@H](O)CN(C)Cc1ccccc1)CC2',
    r'C1=C[C@H]2C[C@@H]1C[C@H]2CN1CC[NH+](C2CCCCCC2)CC1',
    r'COC(=O)c1sccc1C[NH+](C)CC(=O)N(C)C',
    r'CCN(C)S(=O)(=O)c1ccc(C[NH2+]C)cc1F',
    r'NC(=O)N1CCC[C@H](C(=O)N2CCC(C(=O)[O-])CC2)C1',
    r'CCc1ccc(S(=O)(=O)NCC2([NH+](C)C)CCOCC2)s1',
    r'Cc1nn(-c2ccccc2)c(C)c1CN1C[C@@H](C[NH+]2CCCC2)[C@@H](CO)C1',
    r'CC1CCN(c2[nH+]cccc2C(=O)[O-])CC1',
    r'CC(C)CN(CCC#N)C(=O)NC[C@@H]1CC[C@H](C(=O)[O-])O1',
    r'CC(C)N1CC[C@H]([NH2+][C@H](C)CCc2ccc3c(c2)OCO3)C1=O',
    r'Cc1nc(CCC[NH+]2CCC[C@H]2C(N)=O)cs1',
    r'Cc1ccc(C)c([C@@H](O)[C@@H](C)[C@@H](C)C(=O)[O-])c1',
    r'Cc1nn(C)c2ncc(NC(=O)NC[C@H](c3ccccc3)[NH+](C)C)cc12',
    r'NC(=O)c1cccc(CNC(=O)[C@@H]2C[C@@H]3C=C[C@H]2C3)c1',
    r'CN(C(=O)N[C@H]1CC[C@@H]([NH+](C)C)C1)[C@@H]1CCN(c2ccccc2F)C1=O',
    r'O=C(Nc1cccnc1)C1=CC=CN2CCS(=O)(=O)N=C12', r'CCn1ncc(C[NH2+]C)c1C1CC1',
    r'CC(C)[C@@H](ON1C(=O)c2ccccc2C1=O)C(=O)[O-]',
    r'C[C@H]1CCCC[NH+]1C[C@@H]1CCC(C)(C)[C@@H]1[NH3+]',
    r'CNC(=O)CN1CCN(C(=O)c2nc(C)n[nH]2)CC1',
    r'Cc1ccccc1C[NH+](C)CC(=O)Nc1cccc(S(=O)(=O)/N=C2\CCCN2)c1',
    r'C[C@H]1OCC[C@H]1C(=O)NCc1noc(C(C)(C)C)n1', r'CCOc1ncnc(S(=O)(=O)CC)c1N',
    r'CC(=O)c1cn(CCC(=O)N[C@@H]2CCC[NH+](C)C2)c2ccccc12',
    r'COC(=O)C(CC[C@@]1(C)[C@@H](C)CC=C[C@H]1O)C(=O)OC',
    r'CCOc1ccc(NC(=O)c2ccc(N3C(=O)N4C'
    r'CC5=c6ccccc6=[NH+][C@H]5[C@@]4(C)C3=O)cc2)cc1',
    r'O=C(Cn1nnn(-c2cccs2)c1=O)NC[C@@H]1CN(Cc2ccccc2)CCO1',
    r'[O-]c1nc(-c2cccnc2)nc2c1CC[NH+](Cc1ccnc(N3CCOCC3)n1)C2',
    r'Cc1ccn2c(=O)c(C(=O)Nc3n[n-]c(C(F)(F)F)n3)cnc2c1',
    r'O=c1[nH]nc([O-])n1/N=C/c1ccco1', r'N#CCCCNC(=O)[C@H]1CC[C@H](C[NH3+])O1',
    r'NC(=O)[C@@H]1CCCN(C(=O)Cn2nc(-c3cccs3)oc2=O)C1',
    r'CC(C)[C@H](CNC(=O)N[C@H]1C[C@@H]1C)N1CC[NH+](C)CC1',
    r'CCCCCSc1nnc([O-])[nH]c1=O',
    r'Cn1cc[nH+]c1C[C@H]1CCC[NH+](Cc2ncc(-c3ccccc3Cl)o2)C1',
    r'Cc1nc(C)c(C[NH2+]C[C@H](C(C)C)N2CCOCC2)s1',
    r'CC[C@H](C)[NH2+]CCc1n[nH]c(-c2ccco2)n1',
    r'N#CC(C#N)=CNCCCN1C(=O)[C@@H]2[C@@H]3C=C[C@@H](C3)[C@H]2C1=O',
    r'C/[NH+]=C(\NCC(C)(C)c1ccc(OC)cc1)N[C@H]1C[C@H]1C',
    r'[NH3+][C@H]1CCC[C@@H]([NH+]2CC[C@@H](c3ccccc3)C2)C1',
    r'COc1cccc([C@H]2C[NH2+]CC[NH+]2C)c1',
    r'COCCC(=O)NC[C@@H]1CC[C@@H](C(=O)[O-])O1', r'C#CCNCC(=O)NC[C@@H](C)CO',
    r'O[C@@H]1C[C@H](CNCc2cnn(-c3ccccc3)c2)[NH+](Cc2ccccc2)C1',
    r'CN[C@@]1(C(N)=O)CC[C@@H](N2CC[C@H]([NH+]3CCCCC3)C2)C1',
    r'CC(C)CC[NH+]1Cc2cccc(NC(=O)C(=O)NC[C@@H]3CC=CCC3)c2C1',
    r'C[C@H](O)C[C@@H](Cc1cccc(Br)c1)C(=O)[O-]',
    r'C[C@H]1Cc2cc(C(=O)C3=C([O-])C(=O)N(CCN4CCOCC4)[C@H]3c3ccco3)ccc2O1',
    r'C[C@H]1OCC[C@H]1C(=O)N1CCO[C@H](C#N)C1',
    r'N#CC1(C#N)[C@H](C(N)=O)[C@@H]1c1ccc(F)cc1',
    r'COc1cc(NS(=O)(=O)c2ccc(N3C(=O)'
    r'[C@H]4[C@@H]5C=C[C@@H](C5)[C@@H]4C3=O)cc2)nc(OC)n1',
    r'COc1ccc(O[C@H]2CC[C@@]([NH2+]C(C)C)(C(=O)[O-])C2)cc1',
    r'C[C@H]1CCC[C@@H]([NH+](C)CC(=O)N2CCC(C(=O)N3CCCC3)CC2)C1',
    r'Cn1ccc(S(=O)(=O)N2CCO[C@H](CC(=O)[O-])C2)c1',
    r'CC[C@](C)([NH3+])C(=O)N1CCOc2ccccc2C1',
    r'CC(C)[NH+]1CCC(N2CCN(CC3=c4ccccc4=[NH+]C3)C[C@@H]2CCO)CC1',
    r'Cn1c(=O)c(=O)n(CC(=O)N2CCC3(CC2)OCCO3)c2cccnc21',
    r'CC[C@@H]1CN(S(=O)(=O)C(C)C)CC[C@@H]1[NH2+]Cc1cccc(C#N)c1',
    r'C[NH+](C)CCNC(=O)C[C@@H]1C(=O)NCCN1Cc1c(F)cccc1Cl',
    r'Cc1ccc(Cl)cc1N1CCN(C(=O)[C@@H]2CCC[NH2+]2)CC1',
    r'C[C@H]1CC(N)=C(C#N)C1(C#N)C#N', r'CCC[NH2+]CC[NH+](C)CC(C)(C)O',
    r'CCC[C@H](C[NH3+])[C@@]1(O)CCC[NH+](CC)CC1',
    r'CCc1nc(CN2CC[NH+](Cc3cc(Cl)c(OC)c(OC)c3)CC2)no1',
    r'O=C([O-])C[C@@H]1CN(c2ccc([N+](=O)[O-])cc2Cl)CCO1',
    r'CC(C)(C)OC(=O)[C@@H]1N[C@H](C(=O)[O-])C(C)(C)S1',
    r'CCC1CCC(C[NH2+][C@@H]2CCC[C@H]2[C@@H]2CCCC[NH2+]2)CC1',
    r'COc1cc(C[NH2+][C@@H]2C[C@H]3CC[C@]2(C)C3(C)C)ccc1OCC(N)=O',
    r'O[C@@]1(C[NH2+]C2CCN(C3CCCCC3)CC2)CCOC1',
    r'NC(=O)[C@H]1C[NH2+]CCN1c1nc2c(Br)cccn2n1',
    r'COc1ccc2[nH]cc([C@H](C(=O)[O-])[NH+]3CCN(Cc4cccc5ccccc45)CC3)c2c1',
    r'Cc1cc(C)nc(CCNC(=O)C[C@@H]2C(=O)NCC[NH+]2Cc2ccccc2)n1',
    r'Cc1nc(C[C@@H]([NH3+])[C@@H]2CN(C(C)C)CCO2)cs1',
    r'CC[C@@H](CCO)C[NH2+][C@H]1C[C@H](C)c2c(C)ccc(O)c21',
    r'COCCN1[C@@H](C)CN(C(=O)C[NH+](C)C2CC2)C[C@H]1C',
    r'CC(C)C[C@@H](C[NH+](C)C)NC(=O)N1CCC([NH+]2CCCC2)CC1',
    r'C[C@H](CCn1cncn1)[NH2+]Cc1ccn(C2CCCC2)n1',
    r'C[C@@H](CO)C[NH2+]Cc1cn(C)nc1-c1ccccc1',
    r'Cc1c(F)cc(N)cc1S(=O)(=O)NCC(N)=O',
    r'CC(C)[NH+](C[C@@H](C)O)C1C[C@H](C)O[C@@H](C)C1',
    r'O=C(NC1CCCCC1)C1CCN(C2=NC=NC3=NC=N[C@H]32)CC1',
    r'Cc1ccc(C(=O)N[C@H]2CCC[NH2+][C@H]2C)cc1F',
    r'Cc1cn2c([nH+]1)CC[C@H](NS(=O)(=O)N1CCO[C@H](C)C1)C2',
    r'CCC(C)(C)NC(=O)[C@@H](C)Oc1ccc(C[C@H](C)[NH3+])nc1',
    r'CCc1nn(C)cc1NC(=O)C(=O)N1CCN(CC(F)(F)F)[C@@H](C)C1',
    r'CCOC[C@H](O)CSC1COC1', r'CCn1nc(C)c(Cl)c1C[NH2+]CC1(C)COC1',
    r'O=C(CCNc1ccccc1[N+](=O)[O-])N1CCC[C@@H]([NH+]2CCCC2)C1',
    r'COc1cc(OC)cc([C@H]2CC[NH+](CCC(F)(F)F)C2)c1',
    r'C=Cn1cc(C[NH+]2CC[C@@H](CNC(=O)c3ccc(C#N)cc3)C2)cn1',
    r'C=CC[NH+](CC(=O)[O-])[C@@H]1CCC[C@H](C(C)(C)C)CC1',
    r'CCn1cc([C@H]2OCC[C@H]2C[NH2+]Cc2ccc(C)s2)cn1',
    r'COC(=O)[C@H]1CCCCC[C@@H]1NC(=O)Cn1nc2ccccn2c1=O',
    r'[NH3+]Cc1ccccc1CS(=O)(=O)N1CCN2CCC[C@@H]2C1',
    r'O[C@H]1CCCC[C@@H]1[NH+]1CCN(Cc2cnn(-c3ccccc3)c2)CC1',
    r'Fc1ccc2c(c1)[C@H]([NH2+]C1CC[NH+](CCN3CCOCC3)CC1)CC2',
    r'O=C(OC[C@@H]1CC(=O)N(c2ccccc2)C1)[C@@H]1CCC[NH+]1Cc1ccccc1',
    r'O=C1NCCN1[C@H]1CCC[NH+](Cc2cccc(-c3ccncc3)c2)C1',
    r'CN1CCO[C@@H](Cn2cc(C[NH3+])c3cccnc32)C1',
    r'OC1(C(F)(F)F)[C@H]2CC[C@@H]1C[NH+](Cc1ccccc1)C2',
    r'CCCN(CC)c1cc[nH+]c(C(=O)[O-])c1',
    r'CCO[C@@H]1C[C@@H]([NH+](C)CCn2c(=O)oc3cccnc32)C12CCCC2',
    r'CCNS(=O)(=O)c1ccccc1N[C@H](C)C[C@@H]1CCC[NH2+]1',
    r'COc1ccc(/C=C/C2=[NH+]CCN2)cc1',
    r'CO[C@@H](C[NH2+][C@H]1CCN(C[C@@H]2CCCO2)C[C@@H]1C)c1ccc(F)cc1',
    r'CC(C)c1nsc(NC[C@H](C2CC2)[NH+](C)C)n1',
    r'Cc1cccc(C(C)(C)CNC(=O)NC[C@@H](C)C[NH+]2CCN(C)CC2)c1',
    r'CCNS(=O)(=O)[C@@H]1CCN(C(=O)NC2CC=CC2)C1',
    r'COC(=O)[C@@H](N)CC(=O)OC(C)(C)C',
    r'C[C@@]1(c2ccc(C[NH3+])cc2)NC(=O)NC1=O',
    r'CCOC(=O)[C@H](F)[C@@]1(O)CCC[NH+](C(C)C)CC1',
    r'C[C@@H]1CN(S(=O)(=O)[C@@H]2CCC[NH2+]C2)C[C@H]1C',
    r'O=C1NC(=S)NC(=O)C1=CNc1ccc([N+](=O)[O-])cc1O',
    r'C[NH+]1CCN(CCCNC(=O)C(=O)c2ccc(Br)cc2)CC1',
    r'[NH3+][C@@H](CC(=O)[O-])c1cccc2ccccc12',
    r'C[C@@H](C1CC1)[NH+](CC(=O)Nc1nccs1)C1CC1',
    r'C/[NH+]=C(/NCc1ccc([N+]2=CCCC2)cc1)N[C@H]1CC[C@@H](SC)C1',
    r'C[NH2+]C[C@@H]1C[C@@H]1c1c(F)cccc1Cl',
    r'C[C@H]1N=C(CCNC(=O)CCC2=c3ccccc3=[NH+]C2)CS1',
    r'C[C@@H]1CCCC[C@H]1NC(=O)Cn1cnc([N+](=O)[O-])n1',
    r'C[NH+](Cc1c[nH]nc1-c1ccccc1)C[C@@H](O)CN1CCOCC1',
    r'CC(C)=CC[NH+]1CCC2(CC1)CN(CCN1CC[NH+](C(C)C)CC1)C(=O)O2',
    r'C[C@@H]([C@@H](O)c1ccc2ncnn2c1)[N+](=O)[O-]',
    r'CC[NH+](CC)[C@@](C)(CC)[C@@H](C)O', r'CCc1cc(Cn2cc(N)nn2)n(C)n1',
    r'CC[C@H](C)C[NH+](C)[C@@H]1CC[C@@](CO)([NH2+]C)C1',
    r'CC[NH+]1CCC[C@@H]1CN(C)S(=O)(=O)c1cccc(C[NH3+])c1',
    r'CCC[NH+](CCC)[C@@H]1CCC[C@H]([NH2+]C)C1',
    r'CC[C@H](C)[C@H](NC(=O)N1CCCCC1)C(=O)[O-]',
    r'CC[NH+](CC)[C@@H](C)CNC(=O)N1C[C@@H](C)c2ccccc21',
    r'Cn1ncc(SCC(=O)[O-])c(Cl)c1=O', r'Cc1ccc([C@@]23CCC(=O)N2CCC[NH2+]3)cc1',
    r'C[C@H]1C[C@@H](NCCc2nc3cc(F)ccc3n2C)C[NH+]1C',
    r'C[C@@H]1C[NH+](CCN2C(=O)NC(C)(C)C2=O)C[C@@H](C)S1',
    r'CCN(CCO)C(=O)N[C@H]1CC(=O)N(C(C)(C)C)C1',
    r'CCn1nnc2c(=O)n(CC(=O)NC3CC3)cnc21',
    r'CCCC(=O)N[C@@H]1CCC[NH+](Cc2ncccc2C)C1', r'COC[C@](C)(O)C1(C#N)CCCC1',
    r'CCC[C@H](C)N(C)S(=O)(=O)c1ccc(CC(=O)[O-])s1',
    r'CC1(C)[C@H]2OCC[C@@H]2[C@H]1NC(=O)CCNC(=O)C12CC3CC(CC(C3)C1)C2',
    r'CCN(C(=O)Cn1nc2n(c1=O)CCCCC2)[C@H]1CCS(=O)(=O)C1',
    r'Cc1ccc(S(=O)(=O)N[C@H]2C=C[C@@H](C(=O)[O-])C2)cc1F',
    r'CC[NH2+][C@H](Cc1ncnn1CC)[C@@H]1CN(C)CCO1',
    r'CC[NH2+][C@@]1(C(=O)[O-])CC[C@@H](Oc2cccc(OC)c2)C1',
    r'Cn1cc(S(=O)(=O)N2CCN(C(=O)c3ccccc3O)CC2)cc1C(N)=O',
    r'CC[C@H](C)[C@@H]1CCCC[C@H]([NH2+]C)C1',
    r'Cc1ccc(C[NH2+][C@H](C)CN2CCOC2=O)nc1', r'O=C(CS[C@H]1NN=C(C[C@@H]2CCS(=O)'
    r'(=O)C2)O1)C1=c2ccccc2=[NH+][C@@H]1c1ccccc1',
    r'CCOc1cc(CN2CC[NH+]3CCCC[C@@H]3C2)ccc1OC',
    r'CC[NH+](CC)[C@](C)(CC)[C@H](O)c1cscc1Br',
    r'C[NH+]1CCC(NC(=O)c2ncoc2-c2ccccc2)CC1',
    r'CC(C)(C)C(=O)N1CCC[NH+](C[C@H](O)COCc2ccccc2Cl)CC1',
    r'C[C@]12CC[C@H]3[C@@H](CC[C@]4(O)C[C@@H]'
    r'(O)CC[C@]34C=O)[C@@]1(O)CC[C@H]2C1=CC(=O)OC1',
    r'CCC[NH2+][C@@H]1CC[C@@H](C)C[C@@H]1C[NH+]1CCC(CC)(CC)C1',
    r'O=C(CNC(=O)c1ccc[nH]c1=O)Nc1ccon1',
    r'C[NH+]1CCC2(CC1)CNC(=N)N2c1ccc(Cl)c(Br)c1',
    r'C[NH2+]C[C@H](C)C(=O)Nc1snc(C)c1C(=O)[O-]',
    r'CC[NH+](CC)C[C@@H]1CCN(C(=O)Cc2cc3ccccc3[nH]c2=O)C1',
    r'Cc1c(Cl)ccc2c1NC(=O)[C@@]21[C@@H]2C(=O)N'
    r'(c3ccc(F)cc3)C(=O)[C@@H]2[C@@H]2CCC[NH+]21',
    r'Cc1ncccc1NC(=O)C(=O)NCC[NH+](C)C(C)C',
    r'CC[NH+]1CCN(C(=O)C2CC[NH+](Cc3ccccc3F)CC2)CC1',
    r'C[C@H]([NH2+]Cc1nc(Cc2ccccc2)no1)[C@@H](C)n1cccn1',
    r'CC(=O)N[C@@H]1CCCN(C(=O)C[NH+](Cc2ccc(C)s2)C(C)C)C1',
    r'C/[NH+]=C(/NCCc1cc(C)cc(C)c1)NCc1ncc(C)s1',
    r'CC[C@H](C)N1CCN(C(=O)CC(=O)[O-])CC1', r'COC(=O)[C@H](C#N)C(F)(F)F',
    r'CCC(CC)([C@H](O)COC(C)C)[NH+](C)C',
    r'Cc1nn(C)c(C)c1CNC(=O)[C@@H]1C[NH2+]C[C@H]1C',
    r'CC[C@@H]1C[C@@H](C)CC[C@@H]1[NH2+]CCNS(C)(=O)=O',
    r'C[C@@H]1CCCC[C@H]1[NH+]1CCC([NH2+][C@@H]2CCCCC[C@@H]2CO)CC1',
    r'CC1(C)C[NH+](C[C@@H](O)COCc2ccco2)C1(C)C',
    r'CCCC[NH+]1CCC(NC(=O)C2CC=CC2)CC1',
    r'CCn1nc(C)c(CNC(=O)[C@H]2[NH+]=c3ccccc3=C2NC(=O)c2cccc(C)c2)c1C',
    r'COc1ccc(OC)c(S(=O)(=O)n2cc3c(=O)n(C)c(=O)n(C)c3n2)c1',
    r'C[C@H]1CCC[C@@H](C)N1C(=O)[C@@H]1COCCO1',
    r'[NH3+][C@@H](CCCCC(=O)[O-])c1ccc2[nH]c(=O)[nH]c2c1',
    r'C=CC[NH+](CC(=O)[O-])[C@H](C)c1ccc(F)cc1O',
    r'C[C@@]1(C(=O)[O-])CCCCCC[C@@H]1O',
    r'COCCN1C(=O)CC[C@@H]2C[NH+](Cc3cc(C)ccc3C)CC[C@@H]21',
    r'C[NH+](CC(=O)NC(C)(C)C)C1CCC2(CC1)OCCO2',
    r'CN(C(=O)c1nnn[n-]1)[C@H]1CCC[C@H]1C[NH3+]',
    r'COc1ccc(C[NH+]2CC[C@]3(O)CCCC[C@@H]3C2)cc1O',
    r'CC[C@H](C)[C@@H](NC(=O)N1CC(=O)Nc2ccccc21)C(=O)[O-]',
    r'Cc1cc(C)n([C@H]2CCC[NH2+]C2)n1',
    r'CC[NH2+][C@H](Cc1ccccc1Cl)[C@H]1C[NH+](C)CCN1C',
    r'CN(Cc1[nH+]ccn1C)c1cccc(F)c1C(N)=[NH2+]',
    r'Cc1nc(CC(=O)NCCC[NH+]2CCCC[C@H]2C)cs1',
    r'CS(=O)(=O)N1C[C@H](C(N)=O)Oc2ccccc21',
    r'C[C@H]([NH2+]CC(=O)N(C)C)c1ccc(Cl)s1',
    r'C=CCN(CC=C)C[C@@H]1CCC(C)(C)[C@H]1[NH2+]C',
    r'COC[C@H](C)Cc1nnc(C[NH3+])o1',
    r'COC(=O)[C@H]1C[C@H]2CCCC[C@@H]2N1S(=O)(=O)N1CCOCC1',
    r'CC[C@H](C)[NH+](C)CCNC(=O)c1ccc2c(c1)nc(C)n2C',
    r'C[C@@H]1CCC/C(=N/[NH+]=C(/[S-])NCc2ccccc2)C1',
    r'CCOC[C@@H]1CC[NH+](CC(=O)Nc2nc3ccc(F)cc3s2)C1',
    r'C[NH+](CCNC(=O)c1cccn1-c1nnc(N2CCCC2)s1)C1CCCCC1',
    r'CCC(CC)[S@](=O)CCC(=O)[O-]', r'CCOCCCc1cnc(C[NH2+]C)o1',
    r'Cc1cc(CCC(N)=O)cc([C@H]2CCC[NH+]2Cc2cccnc2)n1',
    r'C[NH+](CCc1ccccc1)Cc1c([C@@H]2CCC[NH2+]C2)[nH]c2ncccc12',
    r'NC(=O)C[C@H](NC(=O)[C@@H]1CCC[NH2+]1)C(=O)[O-]',
    r'C[C@@H]1C[C@@H](C)C[NH+](CCNS(=O)(=O)c2ccc(C#N)cc2)C1',
    r'O=C([O-])c1cc(S(=O)(=O)[N-]c2ccc(F)cc2C(=O)[O-])c[nH]1',
    r'C/C=C(\C)[C@@H]1C=C[C@@H]2C[C@H](C)C[C@H](C)'
    r'[C@@H]2[C@@H]1C(=O)C1=C([O-])[C@H](C[C@](C)(O)C(=O)[O-])NC1=O',
    r'CC(C)[C@@H](C)NC(=O)CSCC[NH3+]',
    r'CN(Cc1noc(C2CC2)n1)[C@@H]1CCN(Cc2nnnn2C2CC2)C1',
    r'C/[NH+]=C(/NCCC[NH+]1CCC(C)CC1)NC(C)C',
    r'COCCN(C)C(=O)[C@H]1CCC[NH+]([C@H](C)c2cccc([N+](=O)[O-])c2)C1',
    r'Clc1ccc([C@H](Cc2ccccn2)[NH2+]C[C@H]2C[NH+]3CCN2CC3)cc1',
    r'CNC(=O)CN(C)c1cc(C[NH3+])ccn1',
    r'C[C@H]([NH3+])[C@@H](CC(=O)[O-])c1ccccc1',
    r'Cc1ccc([O-])c(CN2CCN(C(=O)c3ccccc3O)CC2)[nH+]1',
    r'CC[C@@H]1C[NH+]2CCCC[C@@H]2CN1c1cc(C)ccc1F',
    r'O=S(=O)(/N=C(\[O-])c1ccsc1)N1CCCC1',
    r'CCOc1cc(NC(=O)[C@@H]2C=C[C@H]([NH3+])C2)ccc1OC',
    r'CC[NH+]1CCN(Cc2cc(C(=O)NC[C@@H](O)c3ccccc3)no2)CC1',
    r'CC[NH2+][C@@H]1CCCC[C@H]1SC[C@@H](C)CO',
    r'CC[C@H](C)[NH+](C)CC(=O)N1CCNCC1',
    r'CCC(CC)n1ccc(C[C@@H]([NH3+])[C@H]2C[NH+](C)CCN2C)n1',
    r'CC[C@@H]([NH3+])[C@H](c1cc(Br)cs1)N(C)CCS(C)(=O)=O',
    r'Cc1cc(C)c2c(n1)oc1c(N3CC[NH+](CC(N)=O)CC3)ncnc12',
    r'CC[NH+]1CC[C@@H](N(C)C(=O)Nc2cnccc2C)[C@H](C)C1',
    r'O=C(C[NH+]1CCC(C(=O)c2ccc(Cl)cc2)CC1)NC[C@H]1COCCO1',
    r'Cc1ccc(S(=O)(=O)N2CCN(C(=O)[C@H]3CCCC[C@@H]3C(=O)[O-])CC2)cc1C',
    r'CC(C)Nc1cccc(CNC(=O)N[C@@H]2CC[NH+](CC3CC3)C2)c1',
    r'CCc1cc(C(=O)N2CC[C@](O)(CN3CCOCC3)C(C)(C)C2)cc(=O)[nH]1',
    r'CC[NH+](C)CCNC(=O)N1CCC(CC(=O)[O-])CC1',
    r'CCn1c(C)cc(CNCC[NH+]2CCCCC2)c1C',
    r'C[C@H]1CCC[NH+](Cc2ccc(CNC(=O)NNC(=O)C(C)(C)C)cc2)C1',
    r'C/[NH+]=C(/NCc1noc(C(C)(C)C)n1)N[C@@H](C)c1ccc(F)cc1F',
    r'CC(=O)N[C@@H](C(=O)NC1COC1)C(C)C',
    r'C[C@H]1CCCN(C(=O)C2CC[NH+](C[C@@H](O)c3ccc(F)cc3F)CC2)C1',
    r'C[C@@H](C(=O)NCc1cccs1)N1CCn2c(nn(C)c2=O)C1',
    r'C[C@H]1CO[C@H]([C@]2(C)OC[C@H](C)O2)O1',
    r'CN1CCC[NH+](Cc2cc(Cl)ccc2N)CC1',
    r'C[C@@H](C[C@@H]1CCCCC[NH2+]1)[NH2+]CC(C)(C)C',
    r'COC1CC[NH+](CCNC(=O)NCC(C)(C)c2ccncc2)CC1',
    r'COCCNCC1=N[C@H]2C(=N1)C(=O)N(C)C(=O)N2C',
    r'COCCN1C(=O)c2ccccc2[C@@H](C(=O)[O-])[C@H]1c1[nH+]ccn1C',
    r'CNC(=O)c1ccc(O[C@@H]2CCC[C@H]([NH3+])C2)nn1',
    r'C[NH+]1CCC(C[NH+](C)C2CCC(c3ccc(O)cc3)CC2)CC1',
    r'CN(c1ncncc1N)[C@H]1CCSC1', r'SCCCCn1cc[nH+]c1',
    r'Cc1cc2c(cc1C)O[C@@H](C(=O)N(CC[NH+](C)C)Cc1ccco1)C2',
    r'O=C(NCC1([NH+]2CCCCC2)CCOCC1)c1n[nH]c(C2CC2)c1Cl',
    r'NC1=c2cc(Br)ccc2=[NH+]C1', r'CC1(C)CCCC[C@H]1[NH2+]Cc1c[nH]cn1',
    r'CC(C)(CBr)CN1C(=O)C(C)(C)S1(=O)=O',
    r'COc1ccc(S(=O)(=O)NCC[NH2+][C@H]2CCOC3(CCOCC3)C2)cc1',
    r'COC[C@@](C)(O)CNC(=O)C(=O)NCC(C)C', r'CCC1(CC)C(=O)NC(=NNC(N)=S)NC1=O',
    r'c1nnn(C23C[C@H]4C[C@H](CC(c5nc6c7cn[nH]c7ncn6n5)(C4)C2)C3)n1',
    r'CCN[C@H]1CCCC[C@@H]1[NH+](C)Cc1cccc(C)n1',
    r'Cc1ccc(NC(=O)C[NH+](C)C[C@@H](O)CN2CCOCC2)cc1F',
    r'C[C@@H]([NH2+]C[C@H]1CC[C@H](C(N)=O)O1)c1ccc2c(c1)OCCCO2',
    r'CCOC(=O)[C@@H]([NH3+])[C@@H](O)c1ccc(S(C)(=O)=O)cc1',
    r'Cc1ccc(NC(=O)C[NH+]2CCSC(C)(C)C2)c([N+](=O)[O-])c1',
    r'CCC[NH2+][C@H](c1[nH]cc[nH+]1)[C@H]1CCOC2(CCSCC2)C1',
    r'C[NH2+][C@@H](C1CCCC1)[C@@H]1CCc2cccnc21',
    r'COc1ccc(-c2noc(C[NH+]3CCCC[C@@H]3C(=O)[O-])n2)cc1OC',
    r'C[NH2+][C@H](COC(C)C)c1ncccn1',
    r'NC(=O)[C@H]1CCC[NH+](CCCNc2nccc(C(F)(F)F)n2)C1',
    r'C[C@H](CC(=O)N1CCC(C(=O)[O-])CC1)N1C(=O)c2ccccc2C1=O',
    r'CCC[C@@H]1CC[C@H](C[NH3+])[C@H]([NH+]2C[C@@H](C)C[C@@H](C)C2)C1',
    r'CCOCC[NH2+]Cc1c(C)nn(C)c1C',
    r'O=C(C[C@@H]1C[NH2+]CCO1)N[C@H]1C=CS(=O)(=O)C1',
    r'NC(=O)CN1CCN([C@H]2CC(=O)N(CCc3ccccc3)C2=O)CC1',
    r'COCCCn1c(C)c(C)c(C#N)c1NC(=O)C[NH+]1CC(C)(C)C1(C)C',
    r'C[C@@H]1[C@@H](C(=O)[O-])CCN1C(=O)NC[C@@H]1COCCO1',
    r'CC1(C[NH2+][C@@H]2CCc3c(O)cccc32)COC1',
    r'CC(=O)CN1N=C(C)[C@@H](C)n2c1nc1c2c(=O)n(C)c(=O)n1C',
    r'CCc1nc2n(n1)C[C@H]([NH2+]CCNS(=O)(=O)c1ccccc1)CC2',
    r'CC(C)n1cnnc1[C@H]1CCC[NH+](CC2CCC2)C1',
    r'C[C@H](CC[NH+](C)C)NC(=O)Cc1cccc(Cl)c1',
    r'CS(=O)(=O)NCC[NH2+]Cc1ccc(C(=O)[O-])cn1',
    r'COc1ccc(S(=O)(=O)N2CCCN(CC[NH+](C)C)CC2)cc1',
    r'CC1=C([C@H](c2cccc([N+](=O)[O-])c2)c2c(C)n[nH]c2[O-])[C@@H](O)N=N1',
    r'COc1ccc(N2CCn3c2nn(CC(N)=O)c(=O)c3=O)cc1',
    r'C[C@H](NC(=O)NCC1([NH+](C)C)CCCCC1)c1nncn1C',
    r'CCc1ccc(S(=O)(=O)N[C@@H]2CCCC[C@@H]2C[NH+](C)C)cc1',
    r'Cn1cc(C[NH+]2CCN(CCC(N)=O)CC2)c(-c2cccc(Cl)c2)n1',
    r'CCc1cc(C(=O)NCc2cc3n(n2)CCCN(S(C)(=O)=O)C3)on1',
    r'O=C(NNC(=O)[C@H]1C[C@H]2CC[C@@H]1C2)C1=NN=C(c2ccccc2)C1',
    r'CCOc1cc(C[NH+]2CCC[C@H](C(=O)N(CC)CC)C2)ccc1O',
    r'CN(C[C@H]1C=c2ccccc2=[NH+]C1=O)C(=O)c1ccccc1',
    r'CCc1cccc(C)c1NC(=O)C[C@H]1C[NH2+]CCO1',
    r'CC(C)[C@H](C)[NH2+][C@H]1CCOC2(CCCC2)C1',
    r'C[C@@](N)(C(=O)N1CCCC[C@@H]1CCC(=O)[O-])C(F)(F)F',
    r'CC(=O)N1CC[C@@H]([NH2+][C@H](C)c2ccc3[nH]c(=O)[nH]c3c2)C1',
    r'Cc1cc([C@@H]([NH2+]CCN(C)C)C(=O)[O-])ccc1Br',
    r'CCC[NH2+][C@@H](COC)[C@H]1CN(C(C)C)CCO1',
    r'CC(C)Cn1ncc2cc(C(=O)N3CCOC[C@@H]3C(N)=O)cnc21',
    r'CCN(CC)C(=O)CN1CCC([NH+]2CCCCC2)CC1',
    r'C[C@H]([NH+](C)Cc1cnc(Cl)s1)C(C)(C)C',
    r'CCOc1ccc(C[NH+]2CCC[C@H]([C@H](O)c3nccn3C)C2)cc1OC',
    r'C[C@H]1CCCN(C(=O)CS[C@@H]2[NH+]=c3ccc(Cl)cc3=[NH+]2)C1',
    r'CC(=O)N1CCC[C@H](C(=O)N(CC(=O)[O-])CC(F)(F)F)C1',
    r'O=C1NC([O-])=C2C[NH+](Cc3ccco3)CN=C2N1c1ccccc1F',
    r'CO[C@H]1CCCN(C(=O)NCCC[NH+]2CCCCC2)C1',
    r'Cc1ccc(NC(=O)[C@H](C)[NH+](C)Cc2nnc(C3CC3)n2C)c(C)c1',
    r'C[NH+](C)[C@H]1CC[C@H](NC(=O)N2CCN(Cc3ccncc3)CC2)C1',
    r'NC1=C(N)C(=O)c2ncccc2C1=O', r'CC(C)[C@@H](NC(=O)[C@H]1CCCO1)C(=O)[O-]',
    r'CCc1csc([C@H]2CCC[NH+](CC(=O)N(C)OC)C2)n1',
    r'CC(C)c1ccc(CNC(N)=[NH2+])cc1', r'Cc1cc(F)cc(S(=O)(=O)N(C)CC[NH+](C)C)c1',
    r'O=C1C(=O)N(CC[NH+]2CCOCC2)[C@@H](c2cccc([N+](=O)[O-])c2)/C1=C(\O)c1cccs1',
    r'CN(C[C@@H]1CC[NH+](C)C1)C(=O)NCc1ccnc(OCC(F)F)c1',
    r'CN1CCC[NH+](C[C@@H]2CN(C(=O)c3ccc(O)cc3)C[C@@H]2CO)CC1',
    r'CC/[NH+]=C(/NCc1nc(C)no1)N[C@H]1CCN(c2ccccc2)C1',
    r'COc1ccc(C(=O)C2=C([O-])C(=O)N(CC[NH+](C)C)[C@H]2c2ccc(Cl)cc2)cc1Cl',
    r'[NH3+][C@H](CO)c1ccc(N2CCOCC2)c(Cl)c1Cl',
    r'COc1ccc(CNC(=O)CNC(=O)[C@]2(C)CN(S(C)(=O)=O)CC(=O)N2C)cc1',
    r'O=C(C[NH+]1CCC(CO)CC1)NCc1cc(Br)cs1',
    r'C[C@@H](c1nc([C@H]2CSCCO2)no1)N1CC[NH2+]CC1',
    r'C[C@@H](C(=O)N(C)C)[NH+](CC(=O)[O-])C(C)(C)C',
    r'COCC[C@]1(C)O[C@]1(C(=O)OC)C(C)C', r'CCn1c(CC2CC[NH2+]CC2)nn(CCO)c1=O',
    r'CC1(C)CCC[C@]2(C[NH+]=C(N)N2c2cccc(Br)c2)C1',
    r'CCCCC[NH+]1CCN(C(=O)N(C)C)CC1',
    r'CNS(=O)(=O)c1cccc([C@@H](C)[NH2+]C[C@H](C)SC)c1',
    r'Cn1c(=O)c2nc(C[NH+]3CCCCC3)[nH]c2n(C)c1=O',
    r'CCO[C@H]1C(=O)O[C@H]([C@@H](O)CO)C1=O',
    r'CC(C)C[C@H](NC(N)=O)C(=O)N[C@@H](C(=O)[O-])C(C)C',
    r'Cc1n[nH]c(/N=C(\[O-])CNC(=O)c2ccccc2F)n1',
    r'CC[C@H](CSC)N(C)C(=O)[C@@H](C)N(C)c1nccn2cnnc12',
    r'CC1(C)CCC[C@@H](C[NH+](CCO)C2CCCCC2)C1=O',
    r'CC[C@]1(C)NN(c2ccccc2)C([S-])=[NH+]1',
    r'COC(=O)[C@@H]1[C@H](CBr)N1N1C(=O)c2ccccc2C1=O',
    r'CC1=C(C(=O)[O-])N2C(=O)[C@@H](NC(=O)c3c(Br)c(C)nn3C)[C@H]2SC1',
    r'Cc1nc(C[NH+](C)[C@H](C)c2ccc(C(=O)[O-])o2)cs1',
    r'COC(=O)[C@@H](NC(=O)Cn1cnnn1)c1ccc(Cl)c(F)c1',
    r'O=C([O-])[C@H]1CCCN(c2ccc([O-])nn2)C1',
    r'CC[C@H](CC[NH3+])N1CCCN(CC(F)(F)F)CC1',
    r'CCC[NH2+]CC/C=C(/C)[C@@H]1CCOC2(CCSCC2)C1',
    r'O=C(CN1CCN(C(=O)[C@H]2CC(=O)N(c3ccc4c(c3)OCCO4)C2)CC1)N1CCOCC1',
    r'COc1ccc([C@@](C)([NH3+])Cc2[nH+]ccn2C)cc1',
    r'Cc1cscc1C[NH2+]C[C@@H](O)C[NH+]1CCCC1',
    r'COc1cc(Br)ccc1[C@H]([NH3+])C(=O)[O-]',
    r'CCOc1ccc(CN2CC[NH2+][C@H](C(=O)[O-])C2)c(OCC)c1C',
    r'CCn1cc(C[NH+]2CCc3c(F)cc(F)cc3C2)cn1',
    r'O=S(=O)(C1CC1)N1CCC([NH2+]Cc2ccncc2)CC1',
    r'CCCCS(=O)(=O)[N-]c1ccc(NC(=O)[C@H]2CCC[NH+](C)C2)cc1',
    r'Cc1sc(=O)n(CCC(=O)NC2CC(C)(C)[NH2+]C(C)(C)C2)c1C',
    r'O=C(NC[C@H]1CC[C@@H](C(=O)[O-])O1)c1ccc(Br)c(F)c1',
    r'C=CC[NH2+]CC(=O)N[C@H](C)c1c(C)noc1C',
    r'CC1(C)C[C@@H]1NC(=O)[C@@](C)(N)C(F)(F)F',
    r'CCOc1ccccc1/C=C1\Oc2c(ccc([O-])c2C[NH+]2CCN(C)CC2)C1=O',
    r'c1nc(CCN2CCC[NH+]3CCCC[C@@H]3C2)cs1',
    r'COc1ccc(Cc2nc(C)c(CC(=O)[O-])c(=O)[nH]2)cc1OC',
    r'C[C@@H]1C[C@]2(C[NH+]=C(N)N2c2ccc(Cl)cc2)CS1',
    r'C[C@H](O)[C@H](C)[C@H](C(=O)[O-])c1ccccc1Br',
    r'CCCCS[C@H]1CCC[C@@](CO)([NH2+]C(C)C)C1',
    r'C[NH+](C)C1([C@@H](N)c2cnccn2)CCCCCC1',
    r'O=C(N[C@H]1CCS(=O)(=O)C1)C1CC[NH2+]CC1',
    r'N#CCN1CCN(CC(=O)NC(=O)NC2CCCC2)CC1',
    r'CC(C)[C@@H](C)C(=O)Nc1cnn(CC[NH+]2CCCCC2)c1',
    r'[NH3+][C@@H]1CCCCC[C@@H]1c1nnc2c3[nH]cnc3ncn12',
    r'Cc1ccccc1CC[NH+]1CCC[C@@H](C[NH3+])C1',
    r'C[C@@H]1[NH+]=c2ccccc2=C1CCN1C(=O)[C@H]2CCCC[C@@H]2C1=O',
    r'Cc1cc2oc(=O)cc(C[NH+]3CCC[C@@H]3CS(N)(=O)=O)c2cc1C',
    r'COc1cc(OC)cc(OCCN[C@@H]2C[C@H](C)[NH+](C)C[C@H]2C)c1',
    r'CCS[C@H]1CC[C@@H](NC(=O)N(C)CCN(C)C2CC[NH+](C)CC2)C1',
    r'CCOc1cc(C[NH+]2CCN(C3CC3)C(=O)C2)cc(Cl)c1OC',
    r'CC(C)C[C@H](C[NH3+])CN1CC[NH+](CC2CC2)CC1',
    r'C[NH+](CCS(C)(=O)=O)CC(=O)N1CCC[C@H]2CCCC[C@@H]21',
    r'c1cc(CN2CC[NH+](Cc3ccc4c(c3)OCCO4)CC2)no1',
    r'C[C@@H]([NH3+])c1ccccc1O[C@H]1CCO[C@]2(CCSC2)C1',
    r'CCC[NH2+]C1CCC(O)(Cc2nc(C)cs2)CC1',
    r'C[C@@H](C#N)CN(C)C(=O)C1[C@H]2CCC[C@@H]12',
    r'CC(C)OC(=O)[C@H](C)CNC(=O)N[C@H]1CC[C@H]([NH+](C)C)C1',
    r'CCCC12C[C@@H]3C[C@@H](CC([NH3+])(C3)C1)C2',
    r'CNC(=O)[C@@H]1C[NH2+]CCN(C(C)=O)C1',
    r'CN1CCc2cc([C@H](CNC(=O)C(=O)Nc3ccccc3F)[NH+](C)C)ccc21',
    r'Cc1cc(N2CC[C@H](C)[C@H](O)C2)nc(C)[nH+]1',
    r'CCOC(=O)CN1C(=O)CS[C@@]12C(=O)Nc1ccccc12',
    r'C[NH+](C)[C@@H]1CC[C@H](NC(=O)C2CCN(CC(F)(F)F)CC2)C1',
    r'Cc1cc(C)cc(C(=O)N[C@H](C)C(=O)N2CCC3(CC2)[NH2+]CCC2=NC=N[C@@H]23)c1',
    r'Cc1cc([N+](=O)[O-])cnc1Nc1cnn(CC(=O)NCCO)c1',
    r'CCC(CC)(NC(=O)N[C@H]1CCCNC1=O)C(=O)[O-]',
    r'COc1ccc2c(c1)=C[C@H](CN(c1ccc(C)c(C)c1)S(C)(=O)=O)C(=O)[NH+]=2',
    r'C[C@@H](CS(C)(=O)=O)[NH2+][C@H]1CCCOc2c(Br)cccc21',
    r'C[C@H]1CC[C@H](C(=O)[O-])[C@H]([NH+]2CCN3CCC[C@@H]3C2)C1',
    r'O=c1nnc(-c2ccc([N+](=O)[O-])o2)c([O-])[nH]1',
    r'C[C@H]1CC[C@H](C(N)=O)CN1C(=O)Cn1ncc(=O)c2ccccc21',
    r'C[C@H](NC(=O)[C@H](C)N1CC[NH+](CCCO)CC1)c1ccc2c(c1)CCCC2',
    r'CCCN[C@@]1(C#N)CC[C@@H](n2cc[nH+]c2CCC)C1',
    r'CCC[NH2+][C@@H]1CCC[C@H]1CC[NH+]1CCCC(C)(C)CC1',
    r'CCn1cc[nH+]c1[C@@H]1CCCN(C(=O)CSCC[NH+]2CCCC2)C1',
    r'CCOC(=O)C1(C)CC[NH+](C[C@@H](O)c2ccccc2C)CC1',
    r'C[C@H]1C[C@@H](N(C)CC(=O)N2CCOCC2)CC[NH+]1C',
    r'CCCC[C@@H](C)NC(=O)[C@H]1CCC[NH2+][C@@H]1C',
    r'CN(C(=O)N[C@@](C)(C(=O)[O-])C(F)(F)F)c1ccc(F)cc1',
    r'[NH3+][C@@H]1C=C[C@H](C(=O)N2CCC[C@@H]2C(=O)N2CCOCC2)C1',
    r'O=C(c1ccc(F)cc1)[C@H]1CCC[NH+](Cc2c[nH]cn2)C1',
    r'C=C(C)C[NH+]1CCN(CC(=O)N2CCCc3ccccc32)CC1',
    r'CCC(CC)[NH+](C)CCC(=O)NC(N)=O',
    r'CN(C)c1ccc([C@H](CNC(=O)C(=O)Nc2ccccc2C#N)N2CC[NH+](C)CC2)cc1',
    r'CC[NH+](CC)CCN1C(N)=[NH+]C[C@@]12CCCC(C)(C)C2',
    r'COc1ccc(CNC(=O)NC[C@@H](C)[NH+]2CCc3sccc3C2)cn1',
    r'Cn1ncc2c1CCC[C@H]2[NH2+][C@H]1CCN(c2ccc(Cl)cc2)C1=O',
    r'CC1(C)CCCC[C@H]1[NH+]1CCCC[C@@H]1CC[NH3+]',
    r'CCN(Cc1ccccn1)[C@@H]1CCC[C@H]([NH2+]C)C1',
    r'Cc1ccc(C[NH+]2CCN(c3nc4c(c(=O)[nH]c(=O)n4C)n3Cc3cccc(C)c3)CC2)cc1',
    r'CC(C)(C)C[NH+]1CCN(Cc2cccc3cccnc23)C[C@@H]1CCO',
    r'C[NH+]1CCC(NC(=O)C(=O)Nc2ccc(OC3CCCC3)cc2)CC1',
    r'CCC[NH+]1CCCN(Cc2c(Cl)nc3ccccn23)CC1',
    r'O=C(N[C@H]1C=C[C@H](C(=O)[O-])C1)c1cc(F)c(Cl)cc1Cl',
    r'C[C@H]1[C@H](C(=O)[O-])CC[NH+]1CC(=O)NC(C)(C)C',
    r'CNc1nc([C@H]2CCCN(C(=O)CCc3ccccc3)C2)[nH+]c2c1CC[NH+](C)C2',
    r'COCCOc1cccc(C[NH+]2CCC2(C)C)c1', r'Cc1ccc([C@@H](C)NC2=[NH+]CCC2)cc1',
    r'CC(C)NC(=O)NC(=O)[C@@H](C)N1CC[NH+](CCc2ccccc2)CC1',
    r'CC(C)[C@@H](C[NH+](C)C)C(=O)[O-]', r'CN1C[NH+](C)CC2=C1NCNS2(=O)=O',
    r'CCC[NH+](CCC)[C@@H]1CCC(=O)C1',
    r'CC[NH+](CC)[C@H](C)CNC(=O)Nc1ccc2c(c1)NC(=O)[C@H](C)O2',
    r'Cc1ccoc1C[NH+]1CC[C@@H](C)[C@H](C)C1',
    r'CC(C)(C)[NH2+]Cc1ncoc1[C@H]1CCCCO1',
    r'O=C([O-])C12C[C@@H]3C[C@H](C1)CC(n1cc([N+](=O)[O-])cn1)(C3)C2',
    r'CC[C@H](C)[C@H](C)[NH2+]Cc1ncccc1F',
    r'CC1(C)CCC(O)(C[NH2+][C@@H]2CCOC3(CCC3)C2)CC1',
    r'C[C@@H]1CCC[NH+](CCCCNC(=O)Nc2ccccn2)C1',
    r'CC[NH+]1CCC[C@@]2(CC1)C[NH+]=C(N)N2c1ccc(C)cc1',
    r'CC1=C(C(=O)OCC(=O)C2=c3ccccc3=[NH+][C@@H]2C)[C@@H](C)N(C)N1',
    r'CNC(=O)[C@H](C)CN(C)Cc1cc(=O)n2cccc(C)c2[nH+]1',
    r'COc1ccc(Cl)cc1C[C@H]([NH3+])[C@H]1CN2CCC[C@@H]2CO1',
    r'CC[S@](=O)[C@H]1CCCC[C@@H]1NC(=O)NC[C@H](O)c1ccco1',
    r'CCOc1cc2c(cc1OCC)CN(C(=O)NC[C@@H]1CCC[NH+]1CC)CC2',
    r'C[C@@H]1CCO[C@@H]1C(=O)N1CC[C@H](C(N)=O)c2ccccc21',
    r'COC(=O)C[C@H](C)S(=O)(=O)C[C@@H]1CN(C)CCO1',
    r'O=C(NCCC[NH+]1CCCC1)c1ccc2c(c1)NC(=O)[C@@H]1CCCCN21',
    r'C[NH2+][C@@H]1CCC[C@H]([C@@H]2CCC[C@H](S(C)(=O)=O)C2)C1',
    r'N#CCC[NH2+]C1(C(=O)[O-])CC1',
    r'CC(=O)c1cccc(C[NH+]2CC[C@]3(CCC[NH+](Cc4cccc(C)c4)C3)C2)c1',
    r'COC[C@@H](C)NC(=O)N[C@@H](C(N)=O)c1ccccc1',
    r'C[C@@H]1CCC[C@@H]1[NH2+][C@@H]1CCCS[C@@H]1C',
    r'NC(=O)CONC(=O)[C@H]1CCCc2sccc21',
    r'CCn1c(=O)c2ccccc2n2c(CN3CC[C@H](C[NH+](C)CC)C3)nnc12',
    r'CC[C@H]1C[C@H](C)CC[C@@H]1[NH2+][C@@H]1CCN(c2cc(C)nn2C)C1=O',
    r'Cc1ccc(C(=O)NC[C@@H]2C[C@@H](O)C[NH+]2Cc2ccccc2)c(C)n1',
    r'CC(C)CCc1noc(C[NH+](C)[C@H]2CCC[C@@H]2S(C)(=O)=O)n1',
    r'CC(C)[C@@]1(CC2CCOCC2)CCC[NH2+]1',
    r'CC[C@H](NC(=O)c1ccc(C#N)cn1)C(=O)N1CCOCC1',
    r'CCC[NH+]1CCC(N(C)C(=O)NC[C@H]2CCCN(c3ncccn3)C2)CC1',
    r'Cc1nc(-c2cccc(C(=O)N3C[C@@H]4[C@H](C3)C[NH+]3CCCC[C@H]43)c2)n[nH]1'
]


def num_long_cycles(mol):
  """Calculate the number of long cycles.

  Args:
    mol: Molecule. A molecule.

  Returns:
    negative cycle length.
  """
  cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
  if not cycle_list:
    cycle_length = 0
  else:
    cycle_length = max([len(j) for j in cycle_list])
  if cycle_length <= 6:
    cycle_length = 0
  else:
    cycle_length = cycle_length - 6
  return -cycle_length


def penalized_logp(molecule):
  log_p = Descriptors.MolLogP(molecule)
  sas_score = SA_Score.sascorer.calculateScore(molecule)
  cycle_score = num_long_cycles(molecule)
  return log_p - sas_score + cycle_score


class Molecule(molecules_mdp.Molecule):
  """Penalized LogP Molecule"""

  def __init__(self, target_molecule, **kwargs):
    """Initializes the class.

    Args:
      target_molecule: SMILES string. the target molecule against which we
        calculate the similarity.
      **kwargs: The keyword arguments passed to the parent class.
    """
    super(Molecule, self).__init__(**kwargs)
    target_molecule = Chem.MolFromSmiles(target_molecule)
    self._target_mol_fingerprint = self.get_fingerprint(target_molecule)

  def get_fingerprint(self, molecule):
    """Gets the morgan fingerprint of the target molecule.

    Args:
      molecule: Chem.Mol. The current molecule.

    Returns:
      rdkit.ExplicitBitVect. The fingerprint of the target.
    """
    return AllChem.GetMorganFingerprint(molecule, radius=2)

  def get_similarity(self, molecule):
    """Gets the similarity between the current molecule and the target molecule.

    Args:
      molecule: String. The SMILES string for the current molecule.

    Returns:
      Float. The Tanimoto similarity.
    """

    fingerprint_structure = self.get_fingerprint(molecule)

    return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint,
                                          fingerprint_structure)

  def _reward(self):
    molecule = Chem.MolFromSmiles(self._state)
    if molecule is None:
      return -20.0
    sim = self.get_similarity(molecule)
    if sim <= FLAGS.sim_delta:
      reward = penalized_logp(molecule) + 100 * (sim - FLAGS.sim_delta)
    else:
      reward = penalized_logp(molecule)
    return reward * FLAGS.gamma**(self.max_steps - self._counter)


def get_fingerprint(smiles, hparams):
  """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule.
    hparams: tf.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
  """
  if smiles is None:
    return np.zeros((hparams.fingerprint_length,))
  molecule = Chem.MolFromSmiles(smiles)
  if molecule is None:
    return np.zeros((hparams.fingerprint_length,))
  fingerprint = AllChem.GetMorganFingerprintAsBitVect(
      molecule, hparams.fingerprint_radius, hparams.fingerprint_length)
  arr = np.zeros((1,))
  # ConvertToNumpyArray takes ~ 0.19 ms, while
  # np.asarray takes ~ 4.69 ms
  DataStructs.ConvertToNumpyArray(fingerprint, arr)
  return arr


def get_optimized_mols(model_dir, ckpt=80000):
  """Get optimized Molecules.

  Args:
    model_dir: String. model directory.
    ckpt: the checkpoint to load.

  Returns:
    List of 800 optimized molecules
  """
  hparams_file = os.path.join(model_dir, 'config.json')
  with gfile.Open(hparams_file, 'r') as f:
    hp_dict = json.load(f)
    hparams = deep_q_networks.get_hparams(**hp_dict)

  dqn = deep_q_networks.DeepQNetwork(
      input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
      q_fn=functools.partial(
          deep_q_networks.multi_layer_model, hparams=hparams),
      optimizer=hparams.optimizer,
      grad_clipping=hparams.grad_clipping,
      num_bootstrap_heads=hparams.num_bootstrap_heads,
      gamma=hparams.gamma,
      epsilon=0.0)

  tf.reset_default_graph()
  optimized_mol = []
  with tf.Session() as sess:
    dqn.build()
    model_saver = tf.Saver(max_to_keep=hparams.max_num_checkpoints)
    model_saver.restore(sess, os.path.join(model_dir, 'ckpt-%i' % ckpt))
    for mol in all_mols:
      logging.info('Eval: %s', mol)
      environment = molecules_mdp.Molecule(
          atom_types=set(hparams.atom_types),
          init_mol=mol,
          allow_removal=hparams.allow_removal,
          allow_no_modification=hparams.allow_no_modification,
          allow_bonds_between_rings=hparams.allow_bonds_between_rings,
          allowed_ring_sizes=set(hparams.allowed_ring_sizes),
          max_steps=hparams.max_steps_per_episode,
          record_path=True)
      environment.initialize()
      if hparams.num_bootstrap_heads:
        head = np.random.randint(hparams.num_bootstrap_heads)
      else:
        head = 0
      for _ in range(hparams.max_steps_per_episode):
        steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
        valid_actions = list(environment.get_valid_actions())
        observations = np.vstack([
            np.append(
                deep_q_networks.get_fingerprint(act, hparams), steps_left)
            for act in valid_actions
        ])
        action = valid_actions[dqn.get_action(
            observations, head=head, update_epsilon=0.0)]
        environment.step(action)
      optimized_mol.append(environment.get_path())
  return optimized_mol


def main(argv):
  del argv
  model_dir = os.path.join(
      '/namespace/gas/primary/zzp/dqn/r=3/characterization2',
      FLAGS.model_folder)
  all_results = {}
  for i in (0, 2, 4, 6):
    base_dir = os.path.join(model_dir, 'delta_0.%i' % i)
    all_results['delta_0.%i' % i] = get_optimized_mols(base_dir, 800000)

  with gfile.Open(os.path.join(model_dir, 'all_results.json'), 'w') as f:
    json.dump(all_results, f)


if __name__ == '__main__':
  app.run(main)
