import json
import requests
import logging

logging.basicConfig(level=logging.DEBUG)


header = {'Content-Type': 'application/x-www-form-urlencoded'}


# Define question data and build the request dictionary
question_data = {
        "text": "why does my baby have jaundice",
        "userUID": "lorena-trial",
        "lang": "en"
}

# The API endpoint to communicate with
url_post = "https://rosie.umiacs.umd.edu/ask"

# A POST request to the API
post_response = requests.post(url_post, data=question_data, headers=header)

# Print the response
post_response_json = post_response.content
print(post_response_json)