# ProtENN2

ProtENN is an approach for predicting the functional properties of protein
sequences using deep neural networks.

Check back soon for a new paper. This work is also based on the older paper
[Using deep learning to annotate the protein universe](https://www.nature.com/articles/s41587-021-01179-w.epdf?sharing_token=q6tRetZ422gIjtPMP4s5a9RgN0jAjWel9jnR3ZoTv0M6G6LioRgZ9bzThQkXRdrB3jzKxuUul1YK61iQvv0TpiY1g-t8hlEEJAPaWoOEQSPqrFygPoSzQFS2EpxMCyl-LsP8mRRne59fwzepXL22aNjligptda4Cl01WNl1U13I%3D). The main difference is that
in this work,
1. we don't rely on alignment methods to cut up proteins into domains before using 
   the neural networks.
1. the neural networks localize the domain calls within proteins.

You might also be interested in our other related work on 
[ProteInfer](https://google-research.github.io/proteinfer/).

## Usage instructions

If you're interested in the command line interface, see below.

### Install gcloud on your local machine if you don't have it installed
```
sudo apt install -y google-cloud-sdk
gcloud auth login
```

### Create GCP instance with a GPU
```
gcloud compute instances create protenn-gpu --machine-type n1-standard-8 --zone us-west1-b --accelerator type=nvidia-tesla-v100,count=1  --image-family ubuntu-2004-lts --image-project ubuntu-os-cloud --maintenance-policy TERMINATE --boot-disk-size 250
```

### ssh into the machine
```
# You may need to wait ~30 seconds for the machine to boot up first.
gcloud compute ssh protenn-gpu
```

### Install cuda dependencies for GPU support

<!-- disable linter, because we're giving instructions for installing via dpkg 
-->
<!--* pragma: { seclinter_this_is_fine: true } *-->

```
sudo apt update
sudo add-apt-repository ppa:graphics-drivers -y

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb -O /tmp/cuda-keyring_1.0-1_all.deb
sudo dpkg -i /tmp/cuda-keyring_1.0-1_all.deb

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo apt update
sudo apt install -y cuda-10-0 libcudnn7
```

<!--* pragma: { seclinter_this_is_fine: false } *-->

### Install local python virtual environment
```
sudo apt update
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3-venv python3.7 python3-pip python3.7-venv 
mkdir ~/python_venv
cd ~/python_venv
python3.7 -m venv protenn
source ~/python_venv/protenn/bin/activate
cd ~
```

### Get our code from github and install python dependencies (e.g. numpy)
```
sudo apt install -y svn
svn export https://github.com/google-research/google-research/trunk/protenn
pip3 install -r protenn/requirements.txt
```

### Run our code on test sequences
```
python -m protenn.install_models
python -m protenn.predict -i protenn/testdata/test_hemoglobin.fasta -o ~/hemoglobin_predictions.tsv
```
You should see the following output:
```
$ python3 -m protenn.predict -i protenn/testdata/test_hemoglobin.fasta
I0809 16:24:10.495073 140694289487680 protenn.py:420] Running with 1 ensemble elements
I0809 16:24:10.495324 140694289487680 protenn.py:186] Parsing input from protenn/testdata/test_hemoglobin.fasta
Loading models: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.14s/it]
I0809 16:24:12.682420 140694289487680 inference.py:280] Predicting for 1 sequences
Annotating batches of sequences: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.55it/s]
I0809 16:24:13.113105 140694289487680 inference.py:280] Predicting for 1 sequences
Annotating batches of sequences: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.20it/s]

# View your predictions.
$ cat ~/hemoglobin_predictions.tsv

sequence_name	predicted_label	start	end	description
sp|P69891|HBG1_HUMAN	PF00042	25	143	Globin
sp|Q7AP54|HBP2_LISMO	PF05031	31	127	Iron Transport-associated domain
sp|Q7AP54|HBP2_LISMO	PF05031	132	303	Iron Transport-associated domain
sp|Q7AP54|HBP2_LISMO	PF05031	352	488	Iron Transport-associated domain
sp|Q7AP54|HBP2_LISMO	PF00746	528	569	LPXTG cell wall anchor motif

```
