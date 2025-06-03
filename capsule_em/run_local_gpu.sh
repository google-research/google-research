JOB_NAME="norbcapsule_`date +"%b%d_%H%M%S"`"
TAG="norbcapsule_local_gpu"

docker build -f DockerfileGPU -t $TAG $PWD
docker  run -v \
  $HOME/datasets/smallNORB:/root/datasets/smallNORB \
  --runtime=nvidia $TAG \
  --job_name $JOB_NAME
