## Fréchet Deep Speech Distance (FDSD) metrics
Fréchet Deep Speech Distance (FDSD) metrics were originally proposed in
Bińkowski *et al.* "High fidelity speech synthesis with adversarial networks."
arXiv preprint arXiv:1909.11646 (2019).

We provide our re-implementation of these metrics.


## Usage
* Install dependencies

  ```bash
  pip install -m requirements.txt
  ```

* Download and extract the [DeepSpeech 2 checkpoint](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html)
from the Open Seq2Seq library;

* Run

  ```bash
  python3 compute_fdsd.py \
      --sample1 /path/to/real_audio \
      --sample2 /path/to/fake_audio \
      --ds2_ckpt ds2_large/model.ckpt-54800
  ```

