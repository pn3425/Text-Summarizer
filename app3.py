#BERT-based GMM


from flask import Flask, request, render_template
import regex
import torch
import numpy as np
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

# Load the pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Load the BERT-based GMM model 
gmm_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")         #This is a sentence-transformers model: 
  
app = Flask(__name__)

def read_article(data):
    article = data.split(". ")
    sentences = []
    for sentence in article:
        review = regex.sub("[^A-Za-z0-9]", ' ', sentence)
        sentences.append(review.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2):
    tokens1 = tokenizer(sent1, padding=True, truncation=True, return_tensors="pt")
    tokens2 = tokenizer(sent2, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        embeddings1 = model(**tokens1)["last_hidden_state"]
        embeddings2 = model(**tokens2)["last_hidden_state"]

    cosine_sim = torch.nn.functional.cosine_similarity(embeddings1.mean(dim=1), embeddings2.mean(dim=1), dim=1).item()

    return cosine_sim

def build_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(" ".join(sentences[idx1]), " ".join(sentences[idx2]))

    return similarity_matrix

def generate_summary(data, top_n=10):
    summarize_text = []
    sentences = read_article(data)

    # Check if there are enough sentences to generate a summary
    if len(sentences) < top_n:
        return "Not enough sentences to generate a summary."

    # Compute sentence embeddings using BERT-based GMM
    sentence_embeddings = gmm_model.encode([" ".join(sent) for sent in sentences], convert_to_tensor=True)

    # Calculate cosine similarity between sentence embeddings
    similarity_matrix = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings).numpy()

    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i], sentences[i]) for i in range(len(sentences))), reverse=True)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    return ". ".join(summarize_text)

@app.route('/templates', methods=['POST'])
def original_text_form():
    text = request.form['input_text']
    number_of_sent = request.form['num_sentences']
    summary = generate_summary(text, int(number_of_sent))
    return render_template('index1.html', title="Summarizer", original_text=text, output_summary=summary, num_sentences=5)

@app.route('/')
def homepage():
    title = "TEXT summarizer"
    return render_template('index1.html', title=title)

if __name__ == "__main__":
    app.debug = True
    app.run()
