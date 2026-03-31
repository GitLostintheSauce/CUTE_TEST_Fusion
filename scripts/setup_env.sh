#!/bin/bash
# Source this file to set up the CUTE pipeline environment.
# Usage: source scripts/setup_env.sh

export OFT_PATH="$HOME/opt/oft/OpenFUSIONToolkit"
export DYLD_LIBRARY_PATH="$OFT_PATH/bin:$DYLD_LIBRARY_PATH"
export PATH="$OFT_PATH/bin:$PATH"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
