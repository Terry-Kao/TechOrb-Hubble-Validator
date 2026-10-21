FROM debian:12.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates git jq \
      python3 python3-matplotlib python3-numpy python3-pandas python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

