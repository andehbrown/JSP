import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter # for splitting text into chunks
import os, yaml, platform, re
from sentence_transformers import SentenceTransformer
from data import Data # data store module
from retriever import Retriever # retriever module
from generator import Generator # generator module
from evaluator import Evaluator # evaluator module


# --------------- Create text documents of each JSP from corpus

# jsp_corpus = pd.read_pickle('data/corpusstore.pkl')
# count = jsp_corpus.shape[0]

# for i in range(count):
#     filename = "data/jsptxt/" + jsp_corpus['filename'].iloc[i] + ".txt"
#     f = open(filename, "a")
#     text = jsp_corpus['text'].iloc[i]
#     f.write(text)
#     f.close()

# jsp_corpus.to_csv('data/corpus.csv')

# --------------- Create test data

config = yaml.safe_load(open('config.yml'))

if platform.system() == 'Linux':
    VECTOR_ENCODER = SentenceTransformer(config['VECTOR_ENCODER'], device='cuda')
else:
    VECTOR_ENCODER = SentenceTransformer(config['VECTOR_ENCODER'])
    
data = Data(config, VECTOR_ENCODER)
retriever = Retriever(config, VECTOR_ENCODER, data)
generator = Generator(config)

evaluator = Evaluator(config, retriever, generator)

# -------------- Review test data

# test_data = pd.read_pickle('data/eval/testdata.pkl')
# test_data.to_csv('data/eval/testdata.csv')
# contexts = test_data['contexts']
# print(contexts.iloc[0][1])

# --------------- Generate prompt for ChatGPT

# test_data = pd.read_pickle('data/eval/testdata.pkl')
# prompt = "Please extract relevant sentences from the provided context that can potentially help answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase 'Insufficient Information'. While extracting candidate sentences you’re not allowed to make any changes to sentences from given context. "
# context_list = test_data['contexts'].iloc[0]
# context = "".join(str(x) + " " for x in context_list)

# context = re.sub(r'[0-9]+\.', ' ', context) # removes any number followed by a period (numbered lists)
# context = re.sub(r'\s[a-z]\.\s', ' ', context) # removes any single lower case letter followed by a period (numbered lists)
# context = re.sub(r'\"[a-z]\.', ' ', context) # removes any single lower case letter followed by a period (numbered lists)

# num_context_sentences = context.count('. ')
# question = test_data['question'].iloc[0]
# model_input = prompt + "Context: " + context + "Question: " + question
# # print(model_input)

# print(num_context_sentences)

# relevant_context = "To decommission a bulk fuel installation, units must consult with their TLB for agreement. FGSR should be informed, and advice may be sought initially. The installation can exist in three states: In Use, Temporarily Decommissioned, or Decommissioned. If a unit decides to permanently decommission, a program of decommissioning and removal, along with a Land Quality Assessment (LQA), is required. The works budget reflects costs and completion dates. Decommissioning is an irreversible engineering process leaving the asset in a safe state. The specific asset project activity needs to decide whether the asset will be demolished in a short timeframe or decommissioned for the long term. Planning, authorization, and execution of stakeholder actions, associated costs, and detailed work are paramount in transitioning the asset."
# num_relevant_context_sentences = relevant_context.count('. ')
# print(num_relevant_context_sentences)
# print(num_relevant_context_sentences / num_context_sentences)

# ------------------ RAGAS eval

# evaluator.evaluate_with_ragas()

# faithfulness_result = pd.read_pickle('data/eval/faithfulness.pkl')
# answer_relevancy_result = pd.read_pickle('data/eval/answer_relevancy.pkl')
# context_relevancy_result = pd.read_pickle('data/eval/context_relevancy.pkl')

# ------------------ Return similarity scores

# questions = pd.read_csv('eval/questions.csv')
# questions = questions['question'].dropna()
# max_distance = 0.85
# max_distance_question = ""

# for i, question in enumerate(questions):
#     contexts, references = retriever.get_nearest_neighbours(question)
#     count = 0
#     for reference in references:
#         if reference[2] > max_distance:
#             count += 1
#             # max_distance = reference[2]
#             # max_distance_question = question
#             # print(question, reference[1], reference[0])
#     if count > 0:
#         print(question, count)

#  ------------------ Get page lengths

page_finder = data.jsp_corpus['page_finder']

docs = 0
total_chars = 0
x0 = 0
x1 = 0
for list in page_finder:
    if len(list) > 0:
        docs += 1
        total_chars = list[-1]
        num_pages = len(list)
        doc_avg_chars = total_chars / num_pages
        total_chars += doc_avg_chars

average_chars = total_chars / docs

print(average_chars)    