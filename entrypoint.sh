#!/bin/bash
set -e

mkdir -p /app/data
touch /app/data/settings.yaml

# 启动 Python 应用
exec python -u main.py
