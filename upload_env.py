import os
import subprocess
from dotenv import dotenv_values

def main():
    print("Reading .env file...")
    
    # Safely load the .env values as a dictionary
    env_vars = dotenv_values(".env")
    
    if not env_vars:
        print("Error: Could not read .env file or it is empty.")
        return

    # Filter out local localhost URLs that won't work in production
    ignored_keys = ["REDIS_URL", "QDRANT_URL"]
    
    set_env_args = []
    for key, value in env_vars.items():
        if key in ignored_keys:
            print(f"Skipping {key} (needs to be configured manually for production)")
            continue
            
        if value is None:
            continue
            
        # Format as KEY=VALUE, escaping any quotes inside the value
        escaped_value = str(value).replace('"', '\\"')
        set_env_args.append(f'{key}="{escaped_value}"')

    if not set_env_args:
        print("No valid environment variables found to upload.")
        return

    # Join them into the space-separated format required by Azure CLI
    env_vars_string = " ".join(set_env_args)

    print("\nExecuting Azure CLI via Docker to upload secrets...\n")
    
    # We use the docker workaround command that worked for you earlier
    command = (
        f'docker run -it -v ~/.azure:/root/.azure mcr.microsoft.com/azure-cli az containerapp update '
        f'--name "ecomm-cust-assist-app" '
        f'--resource-group "ecomm-rg" '
        f'--set-env-vars {env_vars_string}'
    )

    try:
        # Run the command
        subprocess.run(command, shell=True, check=True)
        print("\n✅ Successfully uploaded environment variables to Azure Container Apps!")
        print("Note: Don't forget to manually set your production REDIS_URL and QDRANT_URL in the Azure Portal.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to upload variables. Error code: {e.returncode}")

if __name__ == "__main__":
    main()
