# A dockerfile that clones the research code's github repo, installs all
# required dependencies and runs a test script.
FROM ubuntu

# Install system packages.
RUN apt-get update && apt-get -yq install git curl g++ gnupg2

# Install python.
RUN apt-get install -y python3-dev python3-pip python3-venv

# Set virtual environment.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install research code from github.
RUN git clone --filter=blob:none --no-checkout https://github.com/google-research/google-research.git \
&& cd google-research \
&& git sparse-checkout set --cone \
&& git checkout master \
&& git sparse-checkout set llm_longdoc_interpretability
WORKDIR google-research/llm_longdoc_interpretability

# Install dependencies.
RUN pip install -r requirements.txt

# Execute run.py
ENTRYPOINT ["python", "run.py"]