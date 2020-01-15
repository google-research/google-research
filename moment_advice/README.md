Experiments in estimating frequency moments with advice. This is an ongoing
research project released as open source to allow collaboration outside Google.

The main function currently uses four methods to estimate frequency moments:
1. Our method of sampling according to advice (read from file)
2. Sampling according to perfect advice (the advice is the exact frequencies
   of the elements in the dataset)
3. PPSWOR (ell_1 sampling)
4. ell_2 sampling

The following types of datasets are supported as input (same format is used
for the advice files):
* graph - Each line in the file represents a directed edge ("u v" or "u v time"
    for temporal graphs), and we output estimates for the frequency moments
    of the in degrees and out degrees of nodes. We allow parallel edges, that
    is, if an edge between two nodes appears twice, it contributes 2 to the
    degrees.
* net_traffic - Each line in the file represents a packet ("src_ip src_port
    dst_ip dst_port protocol") and we estimate the frequency moments for
    unordered IP pairs, that is, the frequency for each IP pair is the number of
    packets send between them (in any direction).
* unweighted_elements - Each line in the file is a key and represents an element
    with that key and weight 1.
* weighted - Each line in the file represents a weighted data element and is of
    the form "key weight".

For a description of the command-line arguments, run
python3 -m moment_advice --help

Contact information:
Edith Cohen edith@cohenwang.com
Ofir Geri ofirgeri@cs.stanford.edu
Rasmus Pagh pagh@itu.dk
