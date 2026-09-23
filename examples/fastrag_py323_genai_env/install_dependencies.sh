#!/bin/bash
set -e

# TODO: review dependencies https://datarobot.atlassian.net/browse/BUZZOK-24542
microdnf update -y
microdnf install -y gcc gcc-c++ which \
  nginx \
  tar gzip unzip zip wget vim-minimal nano

chmod -R 707 /var/lib/nginx /var/log/nginx

pip3 install -U pip --no-cache-dir
pip3 install --no-cache-dir wheel setuptools

pip3 install -r requirements.txt \
  --no-cache-dir \
  --upgrade-strategy eager \
  --extra-index-url https://download.pytorch.org/whl/cpu

pip3 install --no-cache-dir datarobot-fastrag
# datarobot-moderations is installed as a dependency of datarobot-fastrag

microdnf upgrade
microdnf clean all

rm -rf dep.constraints
rm -rf requirements.txt
