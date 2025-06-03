set -e
set -x

virtualenv -p python3 smug_saliency/smug
source smug_saliency/smug/bin/activate

pip install -r smug_saliency/requirements.txt
python -m smug_saliency.masking_test
python -m smug_saliency.utils_test
python -m smug_saliency.mnist_models.train_mnist_test
