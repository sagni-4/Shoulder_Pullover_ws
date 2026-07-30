#!/usr/bin/env bash
# Sets up the virtualenv this workspace is built and run with.
#
# Why a venv at all: the shoulder model needs torch/transformers/albumentations,
# which don't exist as ROS/rosdep packages, and the pinned torch build here
# (cu121, not the newer cu128 default) is required for this machine's GTX 1080 Ti
# (Pascal / sm_61 support was dropped from recent PyTorch cu12.8 wheels).
# `carla` is included too so both nodes in this package can run from one
# consistent environment -- see the top-level README for why.
#
# Default (this machine): reuse the already-set-up venv from the standalone
# shoulder_detection_carla.py testing (~/CARLA_0.9.16/PythonAPI/examples/.venv)
# via a symlink -- it already has the exact pinned torch/transformers/carla
# versions, confirmed working together with rclpy/cv_bridge. This avoids a
# redundant multi-GB download of the same packages.
#
# On a machine without that venv (e.g. a fresh checkout elsewhere), this
# script instead builds a fresh one from scratch with pip.
#
# Usage:
#   cd ~/shoulder_detection_ws
#   ./setup_venv.sh
#   source .venv/bin/activate
#   colcon build --symlink-install   # build WHILE the venv is active, see README
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EXISTING_VENV="$HOME/CARLA_0.9.16/PythonAPI/examples/.venv"

if [ -d "$EXISTING_VENV" ] && "$EXISTING_VENV/bin/python3" -c "import torch, carla" 2>/dev/null; then
    echo "Found an existing venv with torch+carla at $EXISTING_VENV -- reusing it via symlink."
    ln -sfn "$EXISTING_VENV" .venv
else
    echo "No reusable venv found at $EXISTING_VENV -- building a fresh one (this downloads several GB)."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip

    # Pinned to match ~/CARLA_0.9.16/PythonAPI/examples/shoulder_detection_requirements.txt
    pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121
    pip install \
        transformers==4.46.3 \
        albumentations==2.0.8 \
        opencv-python-headless==4.11.0.86 \
        numpy==1.26.4 \
        pillow==12.2.0 \
        carla==0.9.16
fi

echo ""
echo "venv ready at $(pwd)/.venv"
echo "Next: source .venv/bin/activate && colcon build --symlink-install"
