# Deployment

## Local Kubernetes (tested)

Docker Desktop has a built-in Kubernetes cluster (free, no cloud cost).
This was used to verify the manifests actually work, not just that
they parse.

```bash
# 1. Build the image locally (already tagged comorbid-api from earlier)
docker build -t comorbid-api .

# 2. Create the secret holding the API key (never committed)
kubectl create secret generic comorbid-secrets \
  --from-literal=gemini-api-key="$GEMINI_API_KEY"

# 3. Apply the manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 4. Check pods are running
kubectl get pods

# 5. Reach the service
kubectl port-forward service/comorbid-api-service 8000:8000
```

Then `http://localhost:8000/docs` should work the same as the plain
Docker run did in v3 -- the difference is Kubernetes is now managing
2 replicas and can restart a pod if it crashes.

## Azure (planned, not deployed)

This hasn't been deployed to Azure -- the project doesn't need cloud
hosting cost right now. The plan, if/when it does:

1. **Azure Container Registry (ACR)** -- push the same `comorbid-api`
   image built for local Docker/Kubernetes.
2. **Azure Kubernetes Service (AKS)** -- apply the same
   `k8s/deployment.yaml` and `k8s/service.yaml` files used locally.
   This is the point of testing on local Kubernetes first: the
   manifests don't change, only the cluster they're applied to does.
3. **Azure Key Vault** -- replace the local `kubectl create secret`
   step with a Key Vault-backed secret, so the API key isn't stored
   in cluster config directly.
4. **Managed PostgreSQL (Azure Database for PostgreSQL)** with the
   pgvector extension, replacing the local `pgvector/pgvector` Docker
   container from v3.

None of this requires rewriting the application code -- FastAPI,
the Dockerfile, and the Kubernetes manifests stay the same. Only the
infrastructure they run on changes.
