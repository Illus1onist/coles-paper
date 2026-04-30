FROM nvcr.io/nvidia/pytorch:24.04-py3

# Java for PySpark (make-datasets-spark.sh)
RUN apt-get update \
    && apt-get install -y openjdk-21-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r /workspace/requirements.txt \
    && rm /workspace/requirements.txt

# Install the repo itself (embeddings_validation module)
ADD . /home/coles
RUN pip install --no-cache-dir -e /home/coles

ENV _JAVA_OPTIONS="-Xmx32g"

WORKDIR /home/coles
