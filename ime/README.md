
# IME: Interpretable Mixture of Experts

IME consists of a group of interpretable experts and an assignment module which puts weights on different experts.
The assignment module can either be an interpretable model like a linear model or a black box model like an LSTM.
During inference time each sample is assigned to a single expert.

## Data

The datasets can be obtained and put into `data/` folder in the following way:
* [Electricity dataset] in "data/ECL" run `process_ecl.py`
* [Rossmann dataset] download train.csv and store.csv from https://www.kaggle.com/c/rossmann-store-sales/data insert dataset in folder "data/Rossmann" and than run `process_rossmann.py` in "data/Rossmann"


## Requirements

- Python 3.6
- numpy == 1.21.4
- pandas == 1.3.5
- scipy == 1.7.3
- scikit-learn == 1.0.1
- torch == 1.4.0
- torchvision == 0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
- matplotlib == 3.5.1

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```
## Usage
- Commands for training and testing the model IME with white box assignment module and black box assignment module respectively on Rossman:

```bash
python main.py --model IME_WW --data Rossmann --num_experts 20 --learning_rate 0.0001 --learning_rate_gate 0.001 --utilization_hp 1 --smoothness_hp 0.01 --diversity_hp 0

python main.py --model IME_BW --data Rossmann --num_experts 20 --learning_rate 0.0001 --learning_rate_gate 0.001 --utilization_hp 1 --smoothness_hp 0.01 --diversity_hp 0

```
- Scripts for Rossmann and electricity experiments can be found in `scripts/`
- To run Rossmann experiments use
```
 bash scripts/Rossman.sh
```


This is not an officially supported Google product.
