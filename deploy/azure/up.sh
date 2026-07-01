#!/usr/bin/env bash
# Create (or start) the Track B A10 GPU VM with auto-deallocate-on-idle baked in.
#
#   ./up.sh            # create if absent, else just start an existing (deallocated) VM
#
# Override any of these via env:  RG=foo LOC=westus3 ./up.sh
# Spot (preemptible, cheaper):    LOC=westus3 PRIORITY=Spot ./up.sh
set -euo pipefail

RG="${RG:-tcc-trackb}"
LOC="${LOC:-eastus}"
VM="${VM:-tcc-a10}"
SIZE="${SIZE:-Standard_NV36ads_A10_v5}"   # A10, 24 GB, on-demand
IMAGE="${IMAGE:-microsoft-dsvm:ubuntu-hpc:2204:latest}"  # NVIDIA driver + CUDA preinstalled
DISK_GB="${DISK_GB:-256}"
ADMIN="${ADMIN:-azureuser}"
PRIORITY="${PRIORITY:-Regular}"          # "Spot" for a preemptible (cheaper) VM; Spot draws the regional Low-priority/Spot quota
MAX_PRICE="${MAX_PRICE:--1}"             # Spot price cap USD/hr; -1 = pay up to on-demand, never price-evicted
SHUTDOWN_TIME="${SHUTDOWN_TIME:-0200}"    # daily hard backstop, HHMM UTC; set to "off" to disable
SSH_KEY="${SSH_KEY:-$HOME/.ssh/tcc_a10}" # dedicated keypair for this VM (not your default id_rsa)
CLOUD_INIT="$(dirname "$0")/cloud-init-idle-deallocate.yaml"

# Dedicated key so we never touch ~/.ssh/id_rsa.
if [ ! -f "$SSH_KEY" ]; then
  echo ">> generating dedicated SSH key $SSH_KEY"
  ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "tcc-trackb-a10"
fi

# If the VM already exists, just start it (e.g. after an idle deallocate) and exit.
if az vm show -g "$RG" -n "$VM" >/dev/null 2>&1; then
  echo ">> $VM exists -- starting it"
  az vm start -g "$RG" -n "$VM"
  az vm show -d -g "$RG" -n "$VM" --query '{ip:publicIps,power:powerState}' -o table
  exit 0
fi

echo ">> creating resource group $RG in $LOC"
az group create -n "$RG" -l "$LOC" -o none

# Spot: eviction-policy Deallocate (not Delete) keeps the disk + all provisioning so an
# evicted VM can just be restarted (./up.sh) and resume. Empty for a Regular VM.
SPOT_FLAGS=""
if [ "$PRIORITY" = "Spot" ]; then
  SPOT_FLAGS="--priority Spot --eviction-policy Deallocate --max-price ${MAX_PRICE}"
fi

echo ">> creating $VM ($SIZE, priority=$PRIORITY) with system-assigned identity + idle auto-deallocate"
az vm create \
  -g "$RG" -n "$VM" \
  --size "$SIZE" \
  --image "$IMAGE" \
  --os-disk-size-gb "$DISK_GB" \
  --admin-username "$ADMIN" \
  --ssh-key-values "${SSH_KEY}.pub" \
  --public-ip-sku Standard \
  --assign-identity \
  --custom-data "$CLOUD_INIT" \
  $SPOT_FLAGS \
  -o none

# Grant the VM's managed identity permission to deallocate ITSELF (scoped to this VM only).
PRINCIPAL=$(az vm show -g "$RG" -n "$VM" --query identity.principalId -o tsv)
VM_ID=$(az vm show -g "$RG" -n "$VM" --query id -o tsv)
echo ">> granting Virtual Machine Contributor on the VM to its own identity"
az role assignment create \
  --assignee-object-id "$PRINCIPAL" \
  --assignee-principal-type ServicePrincipal \
  --role "Virtual Machine Contributor" \
  --scope "$VM_ID" -o none

# Daily hard backstop: deallocate at a fixed UTC time even if an open SSH session
# kept the idle monitor from firing. Belt-and-suspenders with the idle timer.
if [ "$SHUTDOWN_TIME" != "off" ]; then
  echo ">> setting daily auto-shutdown backstop at ${SHUTDOWN_TIME} UTC"
  az vm auto-shutdown -g "$RG" -n "$VM" --time "$SHUTDOWN_TIME" -o none
fi

echo ">> verifying the GPU is live"
az vm run-command invoke -g "$RG" -n "$VM" \
  --command-id RunShellCommand --scripts "nvidia-smi --query-gpu=name,memory.total --format=csv" \
  --query 'value[0].message' -o tsv

IP=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)
cat <<EOF

Ready.  ssh -i ${SSH_KEY} ${ADMIN}@${IP}
  Provision once (clone + venv + CUDA torch + model pre-download):
    scp -i ${SSH_KEY} $(dirname "$0")/setup-vm.sh ${ADMIN}@${IP}:~
    ssh -i ${SSH_KEY} ${ADMIN}@${IP} 'HF_TOKEN=hf_xxx bash ~/setup-vm.sh'
  Priority: ${PRIORITY} $([ "$PRIORITY" = Spot ] && echo "(preemptible; eviction deallocates -- restart with ./up.sh to resume)")
  Idle policy: deallocates after 30 min with GPU <5% and no logins.
  Nightly backstop: auto-deallocate at ${SHUTDOWN_TIME} UTC (set SHUTDOWN_TIME=off to disable).
  Tune:   ssh in, edit /etc/idle-deallocate.env (takes effect next 5-min tick)
  Watch:  tail -f /var/log/idle-deallocate.log
  Stop now (manual):  ./down.sh
EOF
