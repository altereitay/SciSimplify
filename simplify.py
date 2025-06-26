import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download('punkt')
nltk.download('punkt_tab')

ner = pipeline("token-classification", model="JonyC/scibert-NER-finetuned-improved")

#************************* PART 2 *******************************
def concat_terms(li):
    final_terms = []
    index = 0
    size = len(li)
    while index < size:
        new_term = {}
        start = li[index]['start']
        new_term['start'] = start
        end = li[index]['end']
        index += 1
        while index < size and end + 1 == li[index]['start']:
            end = li[index]['end']
            index += 1
        new_term['end'] = end
        final_terms.append(new_term)
    return final_terms

def find_terms(sentence):
    terms = ner(sentence)
    final_terms = []
    size = len(terms)
    index = 0

    while index < size:
        ent = terms[index]['entity']
        if ent == 'B-Scns':
            start = terms[index]['start']
            end = terms[index]['end']
            index += 1
            while index < size and terms[index]['entity'] == 'I-Scns' and end == terms[index]['start']:
                end = terms[index]['end']
                index += 1
            final_terms.append({'start': start, 'end': end})
        else:
            index += 1
    return final_terms

#************************* PART 3 *******************************
categories = [
    "Physics", "Chemistry", "Biology", "Space and Astronomy",
    "Medicine and Health", "Computers", "Earth", "Social Sciences",
    "Engineering", "Nanotechnology", "General Science and Tools"
]

def find_context(text, categories=None):
    if not categories:
        categories = ["biology", "physics", "medicine", "engineering", "chemistry"]
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    result = classifier(text, categories)
    return result['labels'][0]

def tokenize_text(text):
    return [sent.strip().replace("\n", " ") for sent in sent_tokenize(text)]

def simplify_sentences(sentences):
    tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("grammarly/coedit-large").to("cuda")
    model.eval()

    simplified = []
    for sentence in sentences:
        prompt = f"Simplify this scientific sentence: {sentence}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        output = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=128, num_beams=5)
        simplified.append(tokenizer.decode(output[0], skip_special_tokens=True))

    return simplified

def summarize_text(sentences, chunk_limit=250):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
    chunks, current, length = [], "", 0
    for s in sentences:
        tokens = len(s.split())
        if length + tokens > chunk_limit:
            chunks.append(current.strip())
            current, length = s + " ", tokens
        else:
            current += s + " "
            length += tokens
    if current.strip():
        chunks.append(current.strip())

    summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]["summary_text"] for chunk in chunks]
    return summaries



def clean_text(text):
    return " ".join(text.strip().split())

def simplify_scientific_article(article_text):
    article_text = clean_text(article_text)
    category = find_context(article_text[:256], categories)

    sentences = tokenize_text(article_text)
    simplified_sentences = simplify_sentences(sentences)
    # summaries = summarize_text(simplified_sentences)

    return {
        "simplified_text": " ".join(simplified_sentences),
    }

text = 'With a new design, the bug-sized bot was able to fly 100 times longer than prior versions. With a more efficient method for artificial pollination, farmers in the future could grow fruits and vegetables inside multilevel warehouses, boosting yields while mitigating some of agriculture’s harmful impacts on the environment. To help make this idea a reality, MIT researchers are developing robotic insects that could someday swarm out of mechanical hives to rapidly perform precise pollination. However, even the best bug-sized robots are no match for natural pollinators like bees when it comes to endurance, speed, and maneuverability. Now, inspired by the anatomy of these natural pollinators, the researchers have overhauled their design to produce tiny, aerial robots that are far more agile and durable than prior versions. The new bots can hover for about 1,000 seconds, which is more than 100 times longer than previously demonstrated. The robotic insect, which weighs less than a paperclip, can fly significantly faster than similar bots while completing acrobatic maneuvers like double aerial flips. The revamped robot is designed to boost flight precision and agility while minimizing the mechanical stress on its artificial wing flexures, which enables faster maneuvers, increased endurance, and a longer lifespan. The new design also has enough free space that the robot could carry tiny batteries or sensors, which could enable it to fly on its own outside the lab. “The amount of flight we demonstrated in this paper is probably longer than the entire amount of flight our field has been able to accumulate with these robotic insects. With the improved lifespan and precision of this robot, we are getting closer to some very exciting applications, like assisted pollination,” says Kevin Chen, an associate professor in the Department of Electrical Engineering and Computer Science (EECS), head of the Soft and Micro Robotics Laboratory within the Research Laboratory of Electronics (RLE), and the senior author of an open-access paper on the new design. Chen is joined on the paper by co-lead authors Suhan Kim and Yi-Hsuan Hsiao, who are EECS graduate students; as well as EECS graduate student Zhijian Ren and summer visiting student Jiashu Huang. The research appears today in Science Robotics.'

res = simplify_scientific_article(text)
