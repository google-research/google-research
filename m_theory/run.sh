# cd to the script's own directory.
thisdir=$(dirname $0)
cd $(readlink -f "${thisdir}")


set -e
set -x

virtualenv -p python3 env
source env/bin/activate

pip3 install -r m_theory_lib/requirements.txt
python3 -m dim4.so8_supergravity_extrema.code.extrema

