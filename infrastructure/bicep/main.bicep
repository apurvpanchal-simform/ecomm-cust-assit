param location string = resourceGroup().location
param environmentName string = 'ecomm-env'
param registryName string = 'ecommregistry88921'

param backendImage string = '${registryName}.azurecr.io/ecomm-backend:latest'
param frontendImage string = '${registryName}.azurecr.io/ecomm-frontend:latest'

param backendAppName string = 'ecomm-backend-app'
param frontendAppName string = 'ecomm-frontend-app'

// We accept secrets as secure objects to keep frontend and backend isolated
@secure()
param frontendEnvVars object = {}
@secure()
param backendEnvVars object = {}
param registryServer string = 'ecommregistry88921.azurecr.io'
@secure()
param registryUsername string
@secure()
param registryPassword string

resource environment 'Microsoft.App/managedEnvironments@2023-05-01' existing = {
  name: environmentName
}

resource backendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: backendAppName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      secrets: [
        {
          name: 'registry-password'
          value: registryPassword
        }
      ]
      registries: [
        {
          server: registryServer
          username: registryUsername
          passwordSecretRef: 'registry-password'
        }
      ]
      ingress: {
        external: true // Could be false if only frontend needs access
        targetPort: 8000
      }
    }
    template: {
      containers: [
        {
          name: 'backend'
          image: backendImage
          env: [for item in items(backendEnvVars): {
            name: item.key
            value: item.value
          }]
          resources: {
            cpu: 1
            memory: '2.0Gi'
          }
        }
      ]
    }
  }
}

var frontendEnvArray = [for item in items(frontendEnvVars): {
  name: item.key
  value: item.value
}]

resource frontendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: frontendAppName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      secrets: [
        {
          name: 'registry-password'
          value: registryPassword
        }
      ]
      registries: [
        {
          server: registryServer
          username: registryUsername
          passwordSecretRef: 'registry-password'
        }
      ]
      ingress: {
        external: true
        targetPort: 8501
      }
    }
    template: {
      containers: [
        {
          name: 'frontend'
          image: frontendImage
          env: concat([
            {
              name: 'API_BASE_URL'
              value: 'https://${backendApp.properties.configuration.ingress.fqdn}'
            }
          ], frontendEnvArray)
          resources: {
            cpu: json('0.5')
            memory: '1.0Gi'
          }
        }
      ]
    }
  }
}
