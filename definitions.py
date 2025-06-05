import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))

def simplify_definitions(terms_by_sentence, category, text):
    all_terms = set()
    all_defs = []

    for sentence, terms in terms_by_sentence:
        for t in terms:
            term = sentence[t['start']:t['end']]
            all_terms.add(term)

    for term in all_terms:
        prompt = f"Define the term '{term}' based on the sentence: \"{text}\" in the context of {category}. the definition should be short, no more then 20 words and without bloat"

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