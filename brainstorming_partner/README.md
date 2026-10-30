# Brainstorming Partner

Authors: Aliza Hoffman, Miaomiao Zhang, Brett Beekley

The abstract and paper link are coming soon.

This is not an officially supported Google product

## Getting Started

The brainstorming partner is an agent-based system that proactively listens to a
creative team's brainstorming session and offers suggestions. It consists of:

* A Python server that runs on your machine and manages the agent's behavior
    and memory.
* A web interface that displays the agent's messages and allows users to
    interact with it.
* A Google Cloud project for API access and billing.

Replace `PROJECT_ID` in the links and commands below with your Google Cloud
project ID.

### Set up a New Cloud Project

If you are using a new Cloud project, you need to enable the APIs in that
project first:

1.  [Cloud Speech API Library](https://console.cloud.google.com/apis/library/speech.googleapis.com?project=PROJECT_ID).
1.  [Vertex AI API Library](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project=PROJECT_ID).

### Start the server

1.  Authenticate with Google Cloud Application Default Credentials:

    ```bash
    gcloud auth application-default login
    ```

1.  Run `run.sh serve` to create a virtual environment, install Python
    dependencies from `requirements.txt`, and start the server. `--project` is
    required:

    ```bash
    ./run.sh serve --project=PROJECT_ID
    ```

    You can optionally pass `--chime_sound_path=/path/to/sound.ogg` to override
    the bundled `notification.ogg` chime sound. Run `./run.sh serve --helpshort`
    to list all flags.

1.  On the same machine, open http://localhost:8000 in a browser and allow
    microphone access when prompted. Browsers only allow microphone access on
    `localhost` and HTTPS pages.

### Run the tests

Run `./run.sh` with no arguments to install the dependencies and run the unit
tests. The tests replace Speech-to-Text and Gemini with mocks, so they don't
need a Cloud project or credentials.

## Privacy

While it listens, the brainstorming partner sends microphone audio to Google
Cloud Speech-to-Text and sends the transcript to Gemini on Vertex AI. Both run
in, and are billed to, the Cloud project you pass with `--project`. Tell
everyone in the room that the session is recorded and transcribed, and get
their consent before you start listening. The server keeps transcripts in
memory until it stops, and it includes them in its log messages.
