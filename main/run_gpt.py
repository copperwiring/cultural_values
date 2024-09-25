## GPT4-o code to get agree, disaqgree for a question based on image and it log probability

from openai import OpenAI
 
from PIL import Image
import requests
import io, os 
from io import BytesIO


# Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the image URL and the question you want to ask
image_url = "https://i.postimg.cc/Vkt7TqhY/columbia.png"
question = (
    "Assume you are the person based on demography of the image. What would be the answer to the following question: "
    "'Do you agree with the statement: 'The government should take more responsibility to ensure that everyone is provided for ?' The options are: (A) Strongly agree, (B) Agree, (C) Disagree, (D) Strongly disagree'. My answer in json format is:"

)

# Create the API request
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Use the correct model for your needs
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ],
    max_tokens=300,
    logprobs=True  # Set logprobs to True to get log probabilities
)

# Extract the response from the model
choice = response.choices[0]
output_message = choice.message.content  # Extract the content of the response

# Print the model output
print("Model Output:", output_message)

# Print log probabilities for each token in the response
print("\nLog Probabilities:")
for token_logprob in choice.logprobs.content:
    token = token_logprob.token
    logprob = token_logprob.logprob
    print(f"Token: {token}, LogProb: {logprob}")








