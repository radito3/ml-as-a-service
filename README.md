# ML as a Service

---

## Requirements
- Docker:
  - (Windows/Mac) _Docker Desktop_
  - (Linux) _docker_ CLI, _dockerd_, _containerd_, _runc_
- (optionally) Kubernetes cluster:
  - Either a cluster on a public infrastructure (e.g. GCP, AWS, etc.) or
  - A local cluster, e.g. _minikube_

## How to run

1. Build the docker image with `docker build -t <name>:<tag> .`
2. If you're running a k8s cluster, do the following:
   1. `docker login <registry address> --username <user> --password-stdin`
   2. `docker tag <name>:<tag> <registry>/<name>:<tag>`
   3. `docker push <registry>/<name>:<tag>`
3. If you're running locally: `docker run <name>:<tag> -p 8000:8000`
