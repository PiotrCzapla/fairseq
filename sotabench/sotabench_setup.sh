#!/usr/bin/env bash -x
source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}
REPO="$( cd "$(dirname "$0")" ; pwd -P )"
apt-get install -y git

# required by fairseq
pip install torch
pip install -e .

echo Running temporary setup scripts
SOTABENCH=/workspace/
mkdir -p $SOTABENCH
cd $SOTABENCH
git clone https://github.com/PiotrCzapla/sotabench-eval.git
cd sotabench-eval
git pull
pip install -e .

