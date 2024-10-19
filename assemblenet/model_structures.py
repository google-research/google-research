# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Definitions for AssembleNet/++ structures.

This structure is a `list` corresponding to a graph representation of the
network, where a node is a convolutional block and an edge specifies a
connection from one block to another.

Each node itself (in the structure list) is a list with the following format:
[block_level, [list_of_input_blocks], number_filter, temporal_dilation,
spatial_stride]. [list_of_input_blocks] should be the list of node indexes whose
values are less than the index of the node itself. The 'stems' of the network
directly taking raw inputs follow a different node format:
[stem_type, temporal_dilation]. The stem_type is -1 for RGB stem and is -2 for
optical flow stem. The stem_type -3 is reserved for the object segmentation
input.

In AssembleNet++lite, instead of passing a single `int` for number_filter, we
pass a list/tuple of three `int`s. They specify the number of channels to be
used for each layer in the inverted bottleneck modules.

The structure_weights specify the learned connection weights.
"""


# AssembleNet structure for 50/101 layer models, found using evolution with the
# Moments-in-Time dataset. This is the structure used for the experiments in the
# AssembleNet paper. The learned connectivity weights are also provided.
asn50_structure = [[-1, 4], [-1, 4], [-2, 1], [-2, 1], [0, [1], 32, 1, 1, 0],
                   [0, [0], 32, 4, 1, 0], [0, [0, 1, 2, 3], 32, 1, 1, 0],
                   [0, [2, 3], 32, 2, 1, 0], [1, [0, 4, 5, 6, 7], 64, 2, 2, 0],
                   [1, [0, 2, 4, 7], 64, 1, 2, 0], [1, [0, 5, 7], 64, 4, 2, 0],
                   [1, [0, 5], 64, 1, 2, 0], [2, [4, 8, 10, 11], 256, 1, 2, 0],
                   [2, [8, 9], 256, 4, 2, 0], [3, [12, 13], 512, 2, 2, 0]]
asn101_structure = [[-1, 4], [-1, 4], [-2, 1], [-2, 1], [0, [1], 32, 1, 1, 0],
                    [0, [0], 32, 4, 1, 0], [0, [0, 1, 2, 3], 32, 1, 1, 0],
                    [0, [2, 3], 32, 2, 1, 0], [1, [0, 4, 5, 6, 7], 64, 2, 2, 0],
                    [1, [0, 2, 4, 7], 64, 1, 2, 0], [1, [0, 5, 7], 64, 4, 2, 0],
                    [1, [0, 5], 64, 1, 2, 0], [2, [4, 8, 10, 11], 192, 1, 2, 0],
                    [2, [8, 9], 192, 4, 2, 0], [3, [12, 13], 256, 2, 2, 0]]
asn_structure_weights = [
    [], [], [], [], [], [],
    [[
        0.13810564577579498, 0.8465337157249451, 0.3072969317436218,
        0.2867436408996582
    ]], [[0.5846117734909058, 0.6066334843635559]],
    [[
        0.16382087767124176, 0.8852924704551697, 0.4039595425128937,
        0.6823437809944153, 0.5331538319587708
    ]],
    [[
        0.028569204732775688, 0.10333596915006638, 0.7517264485359192,
        0.9260114431381226
    ]], [[0.28832191228866577, 0.7627848982810974, 0.404977947473526]],
    [[0.23474831879138947, 0.7841425538063049]],
    [[
        0.27616503834724426, 0.9514784812927246, 0.6568767428398132,
        0.9547983407974243
    ]], [[0.5047007203102112, 0.8876819610595703]],
    [[0.9892204403877258, 0.8454614877700806]]
]


# AssembleNet++ structure for 50 layer models, found with the Charades dataset.
# This is the model used in the experiments in the AssembleNet++ paper.
# Note that, in order the build AssembleNet++ with this structure, you also need
# to feed 'object segmentation input' to the network indicated as [-3, 4]. It's
# the 5th block in the architecture.
# If you don't plan to use the object input but want to still benefit from
# peer-attention in AssembleNet++ (with RGB and OF), please use the above
# AssembleNet-50 model instead with assemblenet_plus.py code.
full_asnp50_structure = [[-1, 2], [-1, 4], [-2, 2], [-2, 1], [-3, 4],
                         [0, [0, 1, 2, 3, 4], 32, 1, 1, 0],
                         [0, [0, 1, 4], 32, 4, 1, 0],
                         [0, [2, 3, 4], 32, 8, 1, 0],
                         [0, [2, 3, 4], 32, 1, 1, 0],
                         [1, [0, 1, 2, 4, 5, 6, 7, 8], 64, 4, 2, 0],
                         [1, [2, 3, 4, 7, 8], 64, 1, 2, 0],
                         [1, [0, 4, 5, 6, 7], 128, 8, 2, 0],
                         [2, [4, 11], 256, 8, 2, 0],
                         [2, [2, 3, 4, 5, 6, 7, 8, 10, 11], 256, 4, 2, 0],
                         [3, [12, 13], 512, 2, 2, 0]]
full_asnp_structure_weights = [[], [], [], [], [], [[0.6143830418586731, 0.7111759185791016, 0.19351491332054138, 0.1701001077890396, 0.7178536653518677]], [[0.5755624771118164, 0.5644599795341492, 0.7128658294677734]], [[0.26563042402267456, 0.3033692538738251, 0.8244096636772156]], [[0.07013848423957825, 0.07905343919992447, 0.8767927885055542]], [[0.5008697509765625, 0.5020178556442261, 0.49819135665893555, 0.5015180706977844, 0.4987695813179016, 0.4990265369415283, 0.499239057302475, 0.4974501430988312]], [[0.47034338116645813, 0.4694305658340454, 0.767791748046875, 0.5539310574531555, 0.4520096182823181]], [[0.2769702076911926, 0.8116549253463745, 0.597356915473938, 0.6585626602172852, 0.5915306210517883]], [[0.501274824142456, 0.5016682147979736]], [[0.0866393893957138, 0.08469288796186447, 0.9739039540290833, 0.058271341025829315, 0.08397126197814941, 0.10285478830337524, 0.18506969511508942, 0.23874442279338837, 0.9188644886016846]], [[0.4174623489379883, 0.5844835638999939]]]  # pylint: disable=line-too-long


# AssembleNet++lite structure using inverted bottleneck blocks. By specifing
# the connection weights as [], the model could alos automatically learn the
# connection weights during its training.
asnp_lite_structure = [[-1, 1], [-2, 1],
                       [0, [0, 1], [27, 27, 12], 1, 2, 0],
                       [0, [0, 1], [27, 27, 12], 4, 2, 0],
                       [1, [0, 1, 2, 3], [54, 54, 24], 2, 2, 0],
                       [1, [0, 1, 2, 3], [54, 54, 24], 1, 2, 0],
                       [1, [0, 1, 2, 3], [54, 54, 24], 4, 2, 0],
                       [1, [0, 1, 2, 3], [54, 54, 24], 1, 2, 0],
                       [2, [0, 1, 2, 3, 4, 5, 6, 7], [152, 152, 68], 1, 2, 0],
                       [2, [0, 1, 2, 3, 4, 5, 6, 7], [152, 152, 68], 4, 2, 0],
                       [3, [2, 3, 4, 5, 6, 7, 8, 9], [432, 432, 192], 2, 2, 0]]
asnp_lite_structure_weights = [[], [], [[0.19914183020591736, 0.9278576374053955]], [[0.010816320776939392, 0.888792097568512]], [[0.9473835825920105, 0.6303419470787048, 0.1704932451248169, 0.05950307101011276]], [[0.9560931324958801, 0.7898273468017578, 0.36138781905174255, 0.07344610244035721]], [[0.9213919043540955, 0.13418640196323395, 0.8371981978416443, 0.07936054468154907]], [[0.9441559910774231, 0.9435100555419922, 0.7253988981246948, 0.13498817384243011]], [[0.9964852333068848, 0.8427878618240356, 0.8895476460456848, 0.11014710366725922, 0.6270533204078674, 0.44782018661499023, 0.61344975233078, 0.44898226857185364]], [[0.9970942735671997, 0.7105681896209717, 0.5078442096710205, 0.0951600968837738, 0.624282717704773, 0.8527252674102783, 0.8105692863464355, 0.7857823967933655]], [[0.6180334091186523, 0.11882413923740387, 0.06102970987558365, 0.04484326392412186, 0.05602221190929413, 0.052324872463941574, 0.9969874024391174, 0.9987731575965881]]]  # pylint: disable=line-too-long
