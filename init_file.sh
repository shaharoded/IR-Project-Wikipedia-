#!/bin/bash

set -euxo pipefail

readonly SPARK_JARS_DIR=/usr/lib/spark/jars
readonly SPARK_NLP_VERSION="3.2.1" # Must include subminor version here

# Added 'spacy' to the PIP_PACKAGES array
PIP_PACKAGES=(
  "nltk==3.6.3"
  "spacy"
)
readonly PIP_PACKAGES

mkdir -p ${SPARK_JARS_DIR}

function execute_with_retries() {
  local -r cmd=$1
  for ((i = 0; i < 10; i++)); do
    if eval "$cmd"; then
      return 0
    fi
    sleep 5
  done
  echo "Cmd '${cmd}' failed."
  return 1
}

function download_spark_jar() {
  local -r url=$1
  local -r jar_name=${url##*/}
  curl -fsSL --retry-connrefused --retry 10 --retry-max-time 30 \
    "${url}" -o "${SPARK_JARS_DIR}/${jar_name}"
}

function install_pip_packages() {
  execute_with_retries "pip install ${PIP_PACKAGES[*]}"
}

function install_spark_nlp() {
  download_spark_jar "https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.1-s_2.12/graphframes-0.8.2-spark3.1-s_2.12.jar"
}

# Function to download NLTK data
function install_nltk_data() {
  # Download NLTK data. You can customize this to download only what you need.
  python -m nltk.downloader -d /usr/share/nltk_data all
  # For specific packages, replace 'all' with 'punkt', 'wordnet', etc.
}

# Function to download Spacy language model
function install_spacy_model() {
  python -m spacy download en_core_web_sm
  # Replace 'en_core_web_sm' with the model of your choice, if necessary.
}

function main() {
  # Install Spark Libraries
  echo "Installing Spark-NLP jars"
  install_spark_nlp

  # Install Pip packages
  echo "Installing Pip Packages"
  install_pip_packages

  # Install NLTK data
  echo "Downloading NLTK data"
  install_nltk_data

  # Install Spacy language model
  echo "Downloading Spacy language model"
  install_spacy_model
}

main
