FROM nvcr.io/nvidia/pytorch:24.02-py3

RUN python -c "import torch; print(torch.cuda.is_available())"
RUN python -m pip install --upgrade pip
# ACCELERATE_LOG_LEVEL=info python -m accelerate.commands.launch --config_file deepspeed_zero3.yaml -m act.scripts.run_act --config=gs://madeka-learning-to-clarify/tiny_config_preftest.json

ENV CUDA_HOME=/usr/local/cuda
COPY act/ /act
WORKDIR /act/

COPY /act/src/act/utils/deepspeed_zero3.yaml /act/
COPY /act/src/act/scripts/launch.sh /act/
RUN chmod +x /act/launch.sh

RUN pwd
RUN ls

# Install gcloud.
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install -y google-cloud-cli

# Install Python dependencies.
RUN pip install --no-deps -r requirements.txt
RUN pip install --no-cache \
  absl-py \
  deepspeed==0.12.2 \
  google-generativeai \
  scikit-learn==1.4.0 \
  tokenizers==0.19.1 \
  transformers==4.44.1 \
  trl==0.9.3 \
  wandb==0.17.5
RUN pip install -U datasets
RUN pip install .

RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/

ENTRYPOINT ["./launch.sh"]
