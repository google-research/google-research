# Multi-bitrate compression perceptual evaluation dataset 2022

mucped22: A compression distortion image quality assessment (IQA) database.

_Authors: Luca Versari, George Toderici, Iulia Comșa, Jyrki
Alakuijala, Sami Boukortt, Martin Bruse, Danielle Perszyk_

## Background

Current human evaluated IQA databases, such as
[KADID-10k](http://database.mmsp-kn.de/kadid-10k-database.html),
[TID2013](https://qualinet.github.io/databases/image/tampere_image_database_tid2013/),
[CSIQ](https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/), or
[LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm) are often
used to evaluate the performance of objective image quality metrics, such as
[MS-SSIM](https://www.cns.nyu.edu/pub/eero/wang03b.pdf),
[SSIMULACRA](https://github.com/cloudinary/ssimulacra), or
[NLPD](https://www.cns.nyu.edu/pub/lcv/laparra17-reprint.pdf).

The objective image quality metrics are in turn used to evaluate the quality of
image compression schemes, to minimize the requirements for expensive human
evaluations.

Unfortunately this also creates an extra source of errors when tuning and
comparing image compression schemes.

This makes it necessary to ensure that the first step in this comparison, the
IQA database, suits the requirements of the compression scheme used.

Problems with current IQA databases include using
crowdsourced evaluators potentially without high quality image comparison
experience, comparing images distorted by methods having little or
nothing in common with distortions typically produced by image compression
schemes, and making it hard to differentiate the correlation between objective
metric and human evaluation across different quality levels, e.g. different
distances from visually transparent differences.

## mucped22 evaluations

The mucped22 evaluations were designed by a collaboration of experienced
image compression researchers at Google to fill gaps in the publicly
available sets of IQA databases.

### Source images

The evaluations comprise 22 photographic images composed of people of different
skin tones, landscapes, and close-up pictures of objects and animals, sourced
from [HDR+ Burst Photography Dataset](https://hdrplusdata.org/dataset.html) and
[Unsplash](https://unsplash.com/). The HDR+ images are released under the
[Creative Commons license (CC-BY-SA)](https://creativecommons.org/licenses/by-sa/4.0/)
and the Unsplash images are released under the
[Unsplash license](https://unsplash.com/license).

The images were chosen to roughly represent how images are used on the web, and
were for the same reason downscaled to FullHD resolution, which also reduced the
baseline artefacts of the images.

### Methodology

The evaluations use the methodology from [CLIC](http://compression.cc/).

It consists in requiring a choice between the same crop of two different
distortions of the same image, and computes an
[Elo](https://en.wikipedia.org/wiki/Elo_rating_system)
ranking of distortions based on that. Compared to traditional Opinion
Score methods, it avoids requiring test subjects to calibrate their scores.

The test subject is able to flip between the two distortions, and has the
original image available on the other side for comparison at all times.

The distortions used are encoding and decoding using
[MozJPEG](https://github.com/mozilla/mozjpeg),
[AVIF](https://en.wikipedia.org/wiki/AVIF), and [JPEG
XL](https://en.wikipedia.org/wiki/JPEG_XL) at various settings.

The selection of distortions for an evaluation consists of running a sorting
algorithm on the distortions using the result of an evaluation as the comparison
operator.

Unlike CLIC and to reduce rater effort, the algorithm is not executed
once per image but rather 4 times, every time selecting one image at random.
Simulations of this selection procedure followed by the same ELO computation
used in CLIC shows similar results to running the algorithm once per image.

## Results

The results of the evaluations are located in Google Cloud Storage at
gs://gresearch/mucped22, and contains original images, distorted images, ELO
rankings, and the file `evaluations.json` containing the actual evaluation
results.

To download the results, install [gsutil](https://cloud.google.com/storage/docs/gsutil)
and copy the files:

```console
gsutil -m cp -r gs://gresearch/mucped22 /tmp
```

The JSON file contains a list of objects with the following fields:

- crop: The crop of the original image shown to the evaluator
- greater: The distortion evaluated as closer to the original
- image: The name of the original image
- lesser: The distortion evaluated as further from the original
- random_choice: Whether the evaluator was unable to decide, and picked a random
  distortion
- rater_time_ms: The time spent to evaluate the distortion pair
- image_dims: The dimensions of the original image
- [greater/lesser]_elo: The Elo score in the context of this image for that
  greater and lesser distortions
- [greater/lesser]_[objective metric name]: The [objective metric name] score
  for the greater and lesser distortions
- rater_flips: The number of times the evaluator flipped between the distortions

It is recommended to discard evaluations where:

- The crop isn't fully contained by the original image dimensions due to bugs in
  the rater software, which happened in the order of 15 times
- The rater flipped less than 3 times between the distortions
- The rater spent less than 3000 ms on the evaluation.

### Processing the results

To reproduce the rank per for each distortion, the Rust program
provided in `scripts/elo` can be run:

```console
cd scripts/elo && cargo run --release /tmp/mucped22/evaluations.json && cd -
```

Since we are generally interested in the rank for each distortion per original
image, the script `scripts/elo/create_elos.py` can be used to do that:

```console
cd scripts/elo && python3 create_elos.py -i /tmp/mucped22/evaluations.json -o /tmp/mucped22/elo && cd -
```

To reproduce the objective perceptual image metrics on these results:

- Use the Python script in `scripts/crop.py` to produce the crops for all
  evaluations:

```console
mkdir /tmp/mucped22/crops && python3 scripts/crop.py -i /tmp/mucped22/evaluations.json -o /tmp/mucped22/evaluations.json -id /tmp/mucped22 -od /tmp/mucped22/crops
```

- Check out and build the libjxl repository for the metrics:

```console
git clone https://github.com/libjxl/libjxl.git
cd libjxl
./deps.sh
SKIP_TEST=1 ./ci.sh opt
cd -
```

- Use the Python script in `scripts/meter.py` to compute all the metrics:

```console
python3 scripts/meter.py -ioj /tmp/mucped22/evaluations.json -id /tmp/mucped22/crops -od /tmp/mucped22/crops -md libjxl/tools/benchmark/metrics/ -ed /tmp/mucped22/elo
```
