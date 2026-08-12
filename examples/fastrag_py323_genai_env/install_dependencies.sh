#!/bin/bash
set -e

# Public Maven Central mirror of the DataRobot MLOps jars (groupId: com.datarobot).
# Used instead of DataRobot's internal Artifactory so this image is buildable by
# customers on their own machines with only public-internet access.
# NOTE: not every mlops version is published here — keep DATAROBOT_MLOPS_VERSION on a
# version that exists at https://repo1.maven.org/maven2/com/datarobot/mlops-agent/
MAVEN_CENTRAL_URL="https://repo1.maven.org/maven2/com/datarobot"

# A number of packages here are based on the following custom models image:
# datarobot/dropin-env-base-jdk:ubi8.8-py3.11-jdk11.0.22-drum1.10.20-mlops9.2.8
# (https://github.com/datarobot/datarobot-user-models/blob/master/docker/dropin_env_base_jdk_ubi)
# Downloading MLOps jars prior to build is done via Maven, see pom.xml in the dropin image
# if you need to reproduce the process

# TODO: review dependencies https://datarobot.atlassian.net/browse/BUZZOK-24542
microdnf update -y
microdnf install -y gcc gcc-c++ which \
  java-11-openjdk-headless-1:11.0.25.0.9 java-11-openjdk-devel-1:11.0.25.0.9 \
  nginx \
  tar gzip unzip zip wget vim-minimal nano

chmod -R 707 /var/lib/nginx /var/log/nginx

pip3 install -U pip --no-cache-dir
pip3 install --no-cache-dir wheel setuptools

pip3 install -r requirements.txt \
  --no-cache-dir \
  --upgrade-strategy eager \
  --extra-index-url https://download.pytorch.org/whl/cpu

pip3 install datarobot_fastrag-0.2.1-py3-none-any.whl
# datarobot-moderations is installed as a dependency of datarobot-fastrag

mkdir -p $JARS_PATH
# -f makes curl fail on HTTP errors (e.g. 404) instead of writing an error page into the jar
curl -fSL ${MAVEN_CENTRAL_URL}/datarobot-mlops/${DATAROBOT_MLOPS_VERSION}/datarobot-mlops-${DATAROBOT_MLOPS_VERSION}.jar --output ${JARS_PATH}/datarobot-mlops-${DATAROBOT_MLOPS_VERSION}.jar
curl -fSL ${MAVEN_CENTRAL_URL}/mlops-agent/${DATAROBOT_MLOPS_VERSION}/mlops-agent-${DATAROBOT_MLOPS_VERSION}.jar --output ${JARS_PATH}/mlops-agent-${DATAROBOT_MLOPS_VERSION}.jar

microdnf upgrade
microdnf clean all

rm -rf dep.constraints
rm -rf requirements.txt
rm datarobot_fastrag-0.2.1-py3-none-any.whl
