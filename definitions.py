import os
from dotenv import load_dotenv

from openai import OpenAI

import json
import paho.mqtt.client as mqtt
client = mqtt.Client() # Create MQTT client instance
TOPIC_GO = "articles/simplified"
TOPIC_TERMS = "articles/terms"

load_dotenv()

client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))

def simplify_definitions(terms, category, text):
    all_terms = set(terms)
    all_defs = []


    context = f"I will give you terms to simplifiy based on the text: \"{text}\" in the context of {category}. the definitions should be short, no more then 20 words and without bloat"

    try:
        response = client.chat.completions.create(
            model='gpt-4',
            messages=[{"role": "system", "content": context}],
            temperature=0.3,
            max_tokens=100,
        )
        definition = response.choices[0].message.content.strip()
    except Exception as e:
        definition = f"(error generating definition: {e})"

    for term in all_terms:
        prompt = f"Define the term '{term}'"

        try:
            response = client.chat.completions.create(
                model='gpt-4',
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )
            definition = response.choices[0].message.content.strip()
        except Exception as e:
            definition = f"(error generating definition: {e})"

        all_defs.append((term, definition))

    return all_defs

def load_file_text(file_path):
    with open(file_path, "r") as file:
            article_text = file.read()
    return article_text

def save_simplified_text(file_path, simplified_text):
    with open(file_path, "w") as file:
        file.write(simplified_text)

# Callback when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(TOPIC_TERMS)

# Callback when a PUBLISH message is received from the server
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        status = payload.get("status")
        if status == "new":    
            file_path = payload.get("name") # file_path => full path to the result file
            text = load_file_text(file_path)
            terms = payload.get("terms")
            category = payload.get("category")
            print(f"Received terms from topic '{msg.topic}':\nin category {category} \nwith terms {terms}")
            defs = simplify_definitions(terms, category, text)
            print("Done generating definitions... ")

            client.publish(TOPIC_TERMS, payload=json.dumps({
                'hash': payload.get("hash"),
                'name': file_path,
                "terms" : terms, # terms is list
                "catergory" : category,
                "status" : "done",
                "definitions" : defs,
                }),
                retain=True,
                qos=1)

    except json.JSONDecodeError as e:
        print(f"Invalid JSON received: {e}")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"Error handling message: {e}")




# Attach callbacks
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker
client.connect("localhost", 1883, 60)


# Start the loop to process network traffic and dispatch callbacks
client.loop_forever()