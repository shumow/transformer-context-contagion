#!/usr/bin/env python3
"""
Create, watch and terminate a single RunPod pod that runs deploy/runpod_job.sh unattended,
through the GraphQL API (api.runpod.io) only -- no SSH, no REST host.

Secrets are read from the environment and never printed: RUNPOD_API_KEY (required),
RUNPOD_GH_TOKEN or GH_TOKEN (passed to the pod as GH_TOKEN), HF_TOKEN.

  python deploy/runpod_pod.py balance
  python deploy/runpod_pod.py list
  python deploy/runpod_pod.py create --job smoke --branch <code branch> --results-branch <branch> --max-hours 1.5
  python deploy/runpod_pod.py status <pod id>
  python deploy/runpod_pod.py terminate <pod id>

`create` refuses to start a second pod while one exists (one pod at a time), tries the
COMMUNITY cloud first and falls back to SECURE, and prints the pod id, GPU and $/h.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.request

API = "https://api.runpod.io/graphql"
REPO = "https://github.com/shumow/transformer-context-contagion.git"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU = "NVIDIA L40S"


def gql(query, variables=None):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"])[:800])
    return out["data"]


POD_FIELDS = ("id name desiredStatus costPerHr lastStatusChange machine { gpuDisplayName } "
              "runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } }")


def balance():
    d = gql("query { myself { clientBalance spendLimit } }")["myself"]
    print(f"balance ${d['clientBalance']:.2f}  spend limit {d.get('spendLimit')}")


def list_pods():
    pods = gql("query { myself { pods { " + POD_FIELDS + " } } }")["myself"]["pods"]
    if not pods:
        print("no pods")
    for p in pods:
        print(fmt(p))
    return pods


def fmt(p):
    rt = p.get("runtime") or {}
    up = rt.get("uptimeInSeconds")
    gpus = rt.get("gpus") or []
    util = f" gpu {gpus[0]['gpuUtilPercent']}% mem {gpus[0]['memoryUtilPercent']}%" if gpus else ""
    return (f"{p['id']}  {p.get('name')}  {p.get('desiredStatus')}  "
            f"{(p.get('machine') or {}).get('gpuDisplayName')}  ${p.get('costPerHr')}/h  "
            f"uptime {None if up is None else up // 60}min{util}")


def status(pod_id):
    p = gql("query($id: String!) { pod(input:{podId:$id}) { " + POD_FIELDS + " } }", {"id": pod_id})["pod"]
    print("gone" if p is None else fmt(p))
    return p


def terminate(pod_id):
    gql("mutation($id: String!) { podTerminate(input:{podId:$id}) }", {"id": pod_id})
    print(f"terminate requested for {pod_id}")


def create(args):
    existing = gql("query { myself { pods { id name desiredStatus } } }")["myself"]["pods"]
    if existing and not args.allow_multiple:
        sys.exit(f"refusing: a pod already exists ({existing}); one pod at a time")
    gh = os.environ.get("RUNPOD_GH_TOKEN") or os.environ.get("GH_TOKEN")
    hf = os.environ.get("HF_TOKEN")
    for name, val in (("GH token", gh), ("HF_TOKEN", hf)):
        if not val:
            sys.exit(f"{name} is not set")
    env = [{"key": "TCC_JOB", "value": args.job},
           {"key": "TCC_BRANCH", "value": args.branch},
           {"key": "TCC_RESULTS_BRANCH", "value": args.results_branch},
           {"key": "TCC_MAX_HOURS", "value": str(args.max_hours)},
           {"key": "GH_TOKEN", "value": gh},
           {"key": "HF_TOKEN", "value": hf},
           {"key": "RUNPOD_API_KEY", "value": os.environ["RUNPOD_API_KEY"]},
           {"key": "HF_HUB_ENABLE_HF_TRANSFER", "value": "0"}]
    start = (f"bash -c 'mkdir -p /workspace && git clone -q -b {args.branch} {REPO} /workspace/tcc "
             f"&& bash /workspace/tcc/deploy/runpod_job.sh; "
             f"echo start-command-exit; sleep 120'")
    q = """mutation($input: PodFindAndDeployOnDemandInput) {
             podFindAndDeployOnDemand(input: $input) { id name costPerHr machine { gpuDisplayName } } }"""
    last_err = None
    for cloud in args.clouds:
        inp = dict(cloudType=cloud, gpuCount=1, volumeInGb=0, containerDiskInGb=args.disk_gb,
                   minVcpuCount=4, minMemoryInGb=24, gpuTypeId=GPU, name=f"tcc-{args.job}",
                   imageName=IMAGE, dockerArgs=start, ports="", volumeMountPath="/workspace",
                   env=env, startSsh=False)
        try:
            p = gql(q, {"input": inp})["podFindAndDeployOnDemand"]
            print(f"created {p['id']}  {p['name']}  {cloud}  {p['machine']['gpuDisplayName']}  ${p['costPerHr']}/h")
            return p
        except Exception as e:                      # no stock in that cloud, bad field, ...
            last_err = str(e)
            print(f"{cloud}: {last_err[:300]}", file=sys.stderr)
    sys.exit(f"could not create a pod: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance"); sub.add_parser("list")
    s = sub.add_parser("status"); s.add_argument("pod_id")
    t = sub.add_parser("terminate"); t.add_argument("pod_id")
    c = sub.add_parser("create")
    c.add_argument("--job", required=True, choices=["diag", "smoke", "p1", "e4e5", "followup", "followup2", "followup3"])
    c.add_argument("--branch", required=True, help="code branch the pod clones and runs")
    c.add_argument("--results-branch", required=True, help="branch the pod pushes data/trackC to")
    c.add_argument("--max-hours", type=float, required=True, help="watchdog self-terminate")
    c.add_argument("--disk-gb", type=int, default=120)
    c.add_argument("--clouds", nargs="+", default=["COMMUNITY", "SECURE"])
    c.add_argument("--allow-multiple", action="store_true")
    a = ap.parse_args()
    {"balance": lambda: balance(), "list": lambda: list_pods(), "status": lambda: status(a.pod_id),
     "terminate": lambda: terminate(a.pod_id), "create": lambda: create(a)}[a.cmd]()


if __name__ == "__main__":
    main()
