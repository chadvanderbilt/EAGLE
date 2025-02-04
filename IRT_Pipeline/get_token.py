#%%
# This script retrieves an API token using provided credentials.
# The API URL, username, and password should be adjusted as per your environment.
# 
# **Functions Overview:**
# - get_token: Sends a POST request to the specified API URL and retrieves an authentication token.
#
# **Non-Standard Library Descriptions:**
# - requests: Simplifies sending HTTP requests and handling responses in Python.
#
# **Configuration Notes:**
# - Update the `api_url` with your API endpoint.
# - Ensure you replace the `username`, `password`, and `Authorization` token as needed.
# - Disable SSL warnings for development purposes; enable SSL verification for production use.
#%%

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

def get_token():
    # Disable SSL warnings (not recommended for production)
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

    # Define the API URL, headers, and login credentials
    api_url = "https://your-api-endpoint.com/api/login"  # Replace with your API URL
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': 'Bearer YOUR_STATIC_AUTH_TOKEN'  # Replace with your authorization token if needed
    }
    data = {
        'username': 'your_username',  # Replace with your username
        'password': 'your_password'   # Replace with your password
    }

    # Make the POST request to obtain the token
    response = requests.post(api_url, headers=headers, data=data, verify=False)

    # Handle potential errors in the response
    if response.status_code != 200:
        print(f"Failed to retrieve token: {response.status_code} - {response.text}")
        return None

    # Parse the JSON response and extract the token
    response_json = response.json()
    token = response_json.get('token', {}).get('tokenValue', None)

    if token:
        print(f"The token -{token[:7]}- has been authorized: {token[:7]}")
    else:
        print("Token not found in the response.")

    return token

# Example usage
if __name__ == "__main__":
    get_token()
