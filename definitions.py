import os
from dotenv import load_dotenv
import time

from openai import OpenAI

import json
import paho.mqtt.client as mqtt
mqtt_client = mqtt.Client(client_id='definitions', clean_session=False) # Create MQTT client instance
TOPIC_GO = "articles/simplified"
TOPIC_TERMS = "articles/terms"

load_dotenv()

client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))

ASSISTANT_NAME = "Glossary Helper"
ASSISTANT_MODEL = "gpt-4o-mini"

assistant = client.beta.assistants.create(
    name=ASSISTANT_NAME,
    model=ASSISTANT_MODEL,
    instructions=(
        "You are a glossary helper.\n"
        "For any user term, reply with *only* valid JSON:\n"
        '{"term": "<term>", "definition": "<≤20 words>"}\n'
        "Do not add extra keys, prose, or markdown."
    )
)
print("Assistant ID:", assistant.id)

def start_context_thread(context_text: str, category: str) -> str:
    """
    Creates a thread whose first user message contains the reference text
    (or attaches it as a file). Returns the thread ID for reuse.
    """
    
    first_msg = (
        f"Context category: {category}\n\n"
        "Reference text (keep for all future questions):\n"
        f'"""\n{context_text.strip()}\n"""'
    )

    thread = client.beta.threads.create(
        messages=[{"role": "user", "content": first_msg}]
    )
    return thread.id


def wait_for_run(thread_id, run_id, poll=1):
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status in ("completed", "failed", "requires_action"):
            return run
        time.sleep(poll)


def get_definition(thread_id, term):
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=term
    )
    print(f'defining the term: {term}')

    run = client.beta.threads.runs.create(
        thread_id=thread_id, assistant_id=assistant.id
    )

    wait_for_run(thread_id, run.id)

    msg = client.beta.threads.messages.list(thread_id=thread_id).data[0]
    raw_json = msg.content[0].text.value.strip()

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        print("⚠️  Could not parse JSON:", raw_json)
        return None
    
def simplify_definitions(terms, category, context_text):
    thread_id = start_context_thread(context_text, category)
    all_terms = set(terms)

    results = []
    for t in all_terms:
        d = get_definition(thread_id, t)
        if d is not None:
            results.append(d)
    return results


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
                "catergory" : category,
                "status" : "done",
                "terms" : defs,
                }),
                qos=2)

    except json.JSONDecodeError as e:
        print(f"Invalid JSON received: {e}")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"Error handling message: {e}")




# Attach callbacks
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Connect to broker
mqtt_client.connect("localhost", 1883, 60)


# Start the loop to process network traffic and dispatch callbacks
mqtt_client.loop_forever()
