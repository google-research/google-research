# RS-GNN

This repo contains a reference implementation for the RS-GNN model proposed
in `Tackling Provably Hard Representative Selection via Graph Neural Networks`.


# Usage

For running RS-GNN on Cora, **cd to google_research**. Then, create a virtual environment and activate it:
```bash
python -m venv rsgnn
source rsgnn/bin/activate
```

Then, install the requirements by running:
```bash
pip install -r rs_gnn/requirements.txt
```

Then run the following command:

```bash
python -m rs_gnn.main --data_path='rs_gnn/data/'
```

Alternatively, you can use the following command to run `run.sh` that will do all the above (note that this will automatically create a virtualenv):

```bash
sh rs_gnn/run.sh
```

# Cite RS-GNN

If you find RS-GNN useful in your research, we ask that you cite the following paper:

> Kazemi, S.M., Tsitsulin, A., Esfandiari, H., Bateni, M., Ramachandran, D., Perozzi, B., Mirrokni, V.,
> Tackling Provably Hard Representative Selection via Graph Neural Networks,
> 2022
```
@inproceedings{kazemi2022rsgnn,
     author={Kazemi, Seyed Mehran and Tsitsulin, Anton and Esfandiari, Hossein and Bateni, MohammadHossein and Ramachandran, Deepak and Perozzi, Bryan and Mirrokni, Vahab}
     title={Tackling Provably Hard Representative Selection via Graph Neural Networks},
     year = {2022},
    }
```

# Contact us

For questions or comments about the implementation, please contact [mehrankazemi@google.com](mailto:mehrankazemi@google.com) or [tsitsulin@google.com ](mailto:tsitsulin@google.com ).

# Disclaimer

This is not an official Google product.
