FROM registry.access.redhat.com/ubi9/python-312
USER root
RUN ln -s /usr/bin/dnf /usr/bin/microdnf

COPY requirements.txt requirements.txt

COPY dep.constraints dep.constraints
COPY datarobot_fastrag-0.2.1-py3-none-any.whl datarobot_fastrag-0.2.1-py3-none-any.whl

COPY install_dependencies.sh install_dependencies.sh

ENV ANNOY_COMPILER_ARGS="-D_CRT_SECURE_NO_WARNINGS,-DANNOYLIB_MULTITHREADED_BUILD,-march=x86-64"
ENV DEEPEVAL_TELEMETRY_OPT_OUT="YES"

RUN bash install_dependencies.sh

RUN python3.12 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab'); "

# This ensures that the tiktoken vocabulary artifacts are preloaded to the container's filesystem
# from OpenAI's Azure Blob Storage, and tiktoken does not require egress Internet access to work.
ENV TIKTOKEN_CACHE_DIR=/opt/.tiktoken_cache
RUN python3.12 -c "import tiktoken; tiktoken.get_encoding('cl100k_base').encode('Hello world');" && \
    # Check that the directory exists and is not empty.
    test -n "$( ls -A /opt/.tiktoken_cache/ )"

# Add the user that will run the model
RUN useradd -m -u 1000 envuser && \
    chown -R envuser /opt
USER envuser

# Disable tracking for agent evaluation frameworks
ENV RAGAS_DO_NOT_TRACK=true \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080

# This makes print statements show up in the logs API
ENV PYTHONUNBUFFERED=1

WORKDIR ${CODE_DIR}
COPY ./*.sh ${CODE_DIR}/

ENV WITH_ERROR_SERVER=1
# Uncomment the following line to switch from Flask to uwsgi server
#ENV PRODUCTION=1 MAX_WORKERS=1 SHOW_STACKTRACE=1

ENTRYPOINT ["/opt/code/start_server.sh"]
