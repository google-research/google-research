# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

from typing import Dict, Any

human_eval: Dict[str, Any] = {}

# (2)
baseline_model: Dict[str, Any] = {
    'no_cuda': True,
    'model_file': 'models:controllable_dialogue/convai2_finetuned_baseline',
    'beam_size': 20,
    'batchsize': 1,
    'beam_min_n_best': 10,
}

greedy_model: Dict[str, Any] = {
    'no_cuda': True,
    'model_file': 'models:controllable_dialogue/convai2_finetuned_baseline',
    'beam_size': 1,
    'batchsize': 1,
    'beam_min_n_best': 10,
}

# (3)
pricing_test: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'intrep_word:-1,intrep_2gram:-1,extrep_word:-1,extrep_2gram:-1',
}

# Repetition models round 1
repetition_model_setting05: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-0.5',
}

repetition_model_setting12: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-1.25',
}

repetition_model_setting35: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-3.5',
}

repetition_model_settinginf: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding': 'extrep_2gram:-1e20',
}

repetition_model_setting35_settinginf: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
}

# NIWF INTERESTINGNESS MODELS
# We have two interestingness models.
# bfw means "beam feature weights" control method and ct means "conditional training" control method
# All the interestingness models have repetition control.

interesting_model_ct_setting0: Dict[str, Any] = {
    'no_cuda':
        True,
    'model_file':
        'models:controllable_dialogue/control_avgniwf10b10e',
    'beam_size':
        20,
    'batchsize':
        1,
    'beam_min_n_best':
        10,
    'set_controls':
        'avg_niwf:0',
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
}

interesting_model_ct_setting3: Dict[str, Any] = {
    **interesting_model_ct_setting0,
    'set_controls': 'avg_niwf:3',
}

interesting_model_ct_setting5: Dict[str, Any] = {
    **interesting_model_ct_setting0,
    'set_controls': 'avg_niwf:5',
}

interesting_model_ct_setting7: Dict[str, Any] = {
    **interesting_model_ct_setting0,
    'set_controls': 'avg_niwf:7',
}

interesting_model_ct_setting9: Dict[str, Any] = {
    **interesting_model_ct_setting0,
    'set_controls': 'avg_niwf:9',
}

# comparable to interesting_model_ct_setting0
interesting_model_bfw_setting200: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,niwf:-200',
}

# comparable to interesting_model_ct_setting3
interesting_model_bfw_setting075: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,niwf:0.75',
}

# comparable to interesting_model_ct_setting5
interesting_model_bfw_setting183: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,niwf:1.83',
}

# comparable to interesting_model_ct_setting7
interesting_model_bfw_setting242: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,niwf:2.42',
}

# comparable to interesting_model_ct_setting9
interesting_model_bfw_setting317: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,niwf:3.17',
}

# INQUISITIVENESS MODELS

inquisitive_model_ct_setting00: Dict[str, Any] = {
    'no_cuda':
        True,
    'model_file':
        'models:controllable_dialogue/control_questionb11e10',
    'beam_size':
        20,
    'batchsize':
        1,
    'beam_min_n_best':
        10,
    'set_controls':
        'question:0',
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
}

inquisitive_model_ct_setting01: Dict[str, Any] = {
    **inquisitive_model_ct_setting00,
    'set_controls': 'question:1',
}

inquisitive_model_ct_setting04: Dict[str, Any] = {
    **inquisitive_model_ct_setting00,
    'set_controls': 'question:4',
}

inquisitive_model_ct_setting07: Dict[str, Any] = {
    **inquisitive_model_ct_setting00,
    'set_controls': 'question:7',
}

inquisitive_model_ct_setting10: Dict[str, Any] = {
    **inquisitive_model_ct_setting00,
    'set_controls': 'question:10',
}

# Compared to inquisitive_model_ct_setting10, this removes extrep_2gram control
# (because it blocks questions), and adds beam reordering
# (i.e. given the top 10 candidates from beam search, choose the one which has lowest extrep_2gram).
# This should give much closer to 100% questions.
inquisitive_model_ct_setting10_better: Dict[str, Any] = {
    **inquisitive_model_ct_setting00,
    'set_controls': 'question:10',
    'weighted_decoding': 'extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
    'beam_reorder': 'best_extrep2gram_qn',
}

# RESPONSIVENESS MODELS

responsiveness_model_bfw_setting_minus_10: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,intrep_2gram:-1e20,lastuttsim:-10',
}

responsiveness_model_bfw_setting_00: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,intrep_2gram:-1e20',
}

responsiveness_model_bfw_setting_05: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,intrep_2gram:-1e20,lastuttsim:5',
}

responsiveness_model_bfw_setting_10: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,intrep_2gram:-1e20,lastuttsim:10',
}

responsiveness_model_bfw_setting_13: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,partnerrep_2gram:-1e20,intrep_2gram:-1e20,lastuttsim:13',
}

# NIDF CT INTERESTINGNESS MODELS
# CT buckets 0,2,4,7,9

interesting_nidf_model_ct_setting0: Dict[str, Any] = {
    'no_cuda':
        True,
    'model_file':
        'models:controllable_dialogue/control_avgnidf10b10e',
    'beam_size':
        20,
    'batchsize':
        1,
    'beam_min_n_best':
        10,
    'set_controls':
        'avg_nidf:0',
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
}

interesting_nidf_model_ct_setting2: Dict[str, Any] = {
    **interesting_nidf_model_ct_setting0,
    'set_controls': 'avg_nidf:2',
}

interesting_nidf_model_ct_setting4: Dict[str, Any] = {
    **interesting_nidf_model_ct_setting0,
    'set_controls': 'avg_nidf:4',
}

interesting_nidf_model_ct_setting7: Dict[str, Any] = {
    **interesting_nidf_model_ct_setting0,
    'set_controls': 'avg_nidf:7',
}

interesting_nidf_model_ct_setting9: Dict[str, Any] = {
    **interesting_nidf_model_ct_setting0,
    'set_controls': 'avg_nidf:9',
}

# BFW NIDF INTERESTINGNESS MODELS
# weights -10,-4,4,6,8 (0 is same as repetition_model_setting35_settinginf)

interesting_nidf_model_bfw_setting_minus_10: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:-10',
}

interesting_nidf_model_bfw_setting_minus_04: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:-4',
}

interesting_nidf_model_bfw_setting_04: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:4',
}

interesting_nidf_model_bfw_setting_06: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:6',
}

interesting_nidf_model_bfw_setting_08: Dict[str, Any] = {
    **baseline_model,
    'weighted_decoding':
        'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20,nidf:8',
}

ALL_SETTINGS = """
baseline_model
greedy_model
inquisitive_model_ct_setting00
inquisitive_model_ct_setting01
inquisitive_model_ct_setting04
inquisitive_model_ct_setting07
inquisitive_model_ct_setting10
inquisitive_model_ct_setting10_better
interesting_nidf_model_bfw_setting_04
interesting_nidf_model_bfw_setting_06
interesting_nidf_model_bfw_setting_08
interesting_nidf_model_bfw_setting_minus_04
interesting_nidf_model_bfw_setting_minus_10
interesting_nidf_model_ct_setting0
interesting_nidf_model_ct_setting2
interesting_nidf_model_ct_setting4
interesting_nidf_model_ct_setting7
interesting_nidf_model_ct_setting9
repetition_model_setting05
repetition_model_setting12
repetition_model_setting35
repetition_model_setting35_settinginf
repetition_model_settinginf
responsiveness_model_bfw_setting_00
responsiveness_model_bfw_setting_05
responsiveness_model_bfw_setting_10
responsiveness_model_bfw_setting_13
responsiveness_model_bfw_setting_minus_10
""".strip().split()
