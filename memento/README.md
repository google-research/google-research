# On Catastrophic Interference in Atari 2600 Games
This is the code for the paper
`On Catastrophic Interference in Atari 2600 Games`
by William Fedus*, Dibya Ghosh*, John D. Martin, Marc G. Bellemare, 
Yoshua Bengio, and Hugo Larochelle (2020).

\* Equal contribution

## Running the Code
### Setup
All of the commands below are run from the parent `google_research` directory.
Start a virtualenv with these commands:

```
virtualenv -p python3 .
source ./bin/activate
```

Then install necessary packages:

```
pip install -r memento/requirements.txt
```

### Training
We provide as an example training a Rainbow agent (`rainbow.gin`) but one can 
substitute below commands with `dqn.gin`.
Train the original agent by executing,

```
python -m memento.train_original_agent
  --gin_files=memento/configs/rainbow.gin \
  --base_dir=/tmp/memento/original_agent
```

Once the original agent is trained (several days depending on hardware) and
checkpointed to `/tmp/memento/original_agent` then train the Memento agent with
the following command,

```
python -m memento.train_memento_agent
  --gin_files=memento/configs/rainbow.gin \
  --base_dir=/tmp/memento/memento_agent \
  --original_base_dir=/tmp/memento/original_agent
```
