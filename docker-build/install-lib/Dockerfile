
ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY requirements1.txt .
RUN pip install --no-cache-dir --user -r requirements1.txt

COPY requirements2.txt .
RUN pip install --no-cache-dir --user -r requirements2.txt

RUN apt-get update && apt-get install -y --no-install-recommends pdsh libaio1 libaio-dev pigz nfs-common nfs-kernel-server && rm -rf /var/lib/apt/lists/*
