#!/usr/bin/env bash
# Manually deallocate the Track B VM (stops GPU billing; keeps the disk ~$10/mo).
#   ./down.sh
set -euo pipefail
RG="${RG:-tcc-trackb}"
VM="${VM:-tcc-a10}"
echo ">> deallocating $VM"
az vm deallocate -g "$RG" -n "$VM"
az vm show -d -g "$RG" -n "$VM" --query '{power:powerState}' -o table
