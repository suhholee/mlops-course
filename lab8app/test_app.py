import requests
import json
import sys

def test_prediction(comment):
    url = 'http://127.0.0.1:8000/predict'
    
    # Prepare the data
    data = {'reddit_comment': comment}
    
    # Make the request
    response = requests.post(url, json=data)
    
    # Print the results
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=4)}")
    
    return response.json()

if __name__ == "__main__":
    comment = sys.argv[1] if len(sys.argv) > 1 else "This is a terrible comment and should be removed!"
    print(f"Testing with comment: '{comment}'")
    result = test_prediction(comment)