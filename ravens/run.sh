TASK=$1
AGENT=$2

python demos.py --task=${TASK} --mode=train --n=1000
python demos.py --task=${TASK} --mode=test  --n=100

python train.py --task=${TASK} --agent=${AGENT} --n_demos=1
python train.py --task=${TASK} --agent=${AGENT} --n_demos=10
python train.py --task=${TASK} --agent=${AGENT} --n_demos=100
python train.py --task=${TASK} --agent=${AGENT} --n_demos=1000

python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=1000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=2000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=5000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=10000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=20000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1 --n_steps=40000

python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=1000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=2000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=5000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=10000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=20000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=10 --n_steps=40000

python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=1000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=2000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=5000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=10000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=20000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=100 --n_steps=40000

python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=1000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=2000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=5000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=10000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=20000
python test.py  --task=${TASK} --agent=${AGENT} --n_demos=1000 --n_steps=40000

python plot.py  --task=${TASK} --agent=${AGENT} --n_demos=1
python plot.py  --task=${TASK} --agent=${AGENT} --n_demos=10
python plot.py  --task=${TASK} --agent=${AGENT} --n_demos=100
python plot.py  --task=${TASK} --agent=${AGENT} --n_demos=1000
