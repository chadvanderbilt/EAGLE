#%%
# This script retrieves the file path for a given slide ID from an API.
# The API URL and token should be configured according to your environment.
#
# **Functions Overview:**
# - get_slide_path: Sends a GET request with a slide ID to an API and retrieves the file path if available.
#
# **Non-Standard Library Descriptions:**
# - requests: Simplifies sending HTTP requests and handling responses in Python.
#
# **Configuration Notes:**
# - Replace `api_url` with your specific API endpoint.
# - Ensure the `Authorization` header contains a valid token.
# - SSL warnings are disabled for development; enable SSL verification in production environments.
#%%

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

def get_slide_path(slide_id):
    # Disable warnings for insecure HTTPS requests (not recommended for production)
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    
    # Define the API endpoint with the slide_id as a parameter
    api_url = f"https://your-api-endpoint.com/api/v1/inventory?key={slide_id}"  # Replace with your API URL
    
    # Authorization token should be replaced with a valid token from your environment
    headers = {
        'Authorization': 'Bearer YOUR_AUTH_TOKEN'  # Replace with your token
    }
    
    # Make the GET request
    response = requests.get(api_url, headers=headers, verify=False)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Check if the 'inventories' list is not empty
        if data.get('inventories'):
            # Return the 'path' of the first item in the 'inventories' list
            return data['inventories'][0].get('path', 'Path not found in inventory.')
        else:
            return "No inventory found for the given slide ID."
    else:
        return f"Failed to retrieve data from the API. Status code: {response.status_code}"

# Example usage
if __name__ == "__main__":
    slide_id = 'your_slide_id'  # Replace with the actual slide ID
    path = get_slide_path(slide_id)
    print(path)
#%%
