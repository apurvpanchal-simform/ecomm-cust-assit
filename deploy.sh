#!/bin/bash
set -e

# Parse the .env file and build a JSON object of parameters
# Keys specifically needed by the Chainlit frontend
FRONTEND_KEYS=("SUPABASE_DB_URL" "JWT_SECRET_KEY" "CHAINLIT_AUTH_SECRET")

FRONTEND_ENV_JSON="{"
BACKEND_ENV_JSON="{"

while IFS='=' read -r key value; do
  # Ignore comments and empty lines
  if [[ -n "$key" && "$key" != \#* ]]; then
     # Remove quotes
     value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
     
     # Check if key belongs to frontend
     is_frontend=0
     for fk in "${FRONTEND_KEYS[@]}"; do
         if [[ "$key" == "$fk" ]]; then
             is_frontend=1
             break
         fi
     done

     if [[ $is_frontend -eq 1 ]]; then
         FRONTEND_ENV_JSON="$FRONTEND_ENV_JSON \"$key\": \"$value\","
     fi
     # Backend needs everything else (we just pass all to backend for simplicity)
     BACKEND_ENV_JSON="$BACKEND_ENV_JSON \"$key\": \"$value\","
  fi
done < .env

# Remove trailing comma and close JSON object
FRONTEND_ENV_JSON="${FRONTEND_ENV_JSON%,} }"
BACKEND_ENV_JSON="${BACKEND_ENV_JSON%,} }"

echo "Fetching Azure Container Registry credentials..."
REG_USER="ecommregistry88921"
REG_PASS=$(docker run --rm -v ~/.azure:/root/.azure mcr.microsoft.com/azure-cli az acr credential show -n $REG_USER --query "passwords[0].value" -o tsv)

echo "Deploying Bicep template with partitioned secrets from .env..."

docker run --rm \
  -v ~/.azure:/root/.azure \
  -v "$(pwd)":/workspace \
  -w /workspace \
  mcr.microsoft.com/azure-cli \
  az deployment group create \
  --resource-group ecomm-rg \
  --template-file infrastructure/bicep/main.bicep \
  --parameters frontendEnvVars="$FRONTEND_ENV_JSON" backendEnvVars="$BACKEND_ENV_JSON" registryUsername="$REG_USER" registryPassword="$REG_PASS"

echo "Deployment finished!"
