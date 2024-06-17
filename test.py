import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pyLDAvis.gensim_models
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel

# using spacy to clean our dataset

def load(file):

    df = pd.read_csv(file)
    content = df['content']
    return content
    

def token_filter(token):
    return not (token.is_punct | token.is_space | token.is_stop | 
                len(token.text) <= 2 | token.like_url | 
                token.like_num | token.like_email)

# will need to add a function to remove: leftover numbers, months, years,
# foreign languages, spam (advertisements), login/account vocabulary

def clean(content):
    nlp = spacy.load("en_core_web_sm") # could exclude things like
    # tagger, ner, etc. 
    
    filtered_tokens = [] 
    # uses list() for batch procesing
    for i in list(nlp.pipe(content)):
        tokens = [token.lemma_.lower() for token in i if
    token_filter(token)]
        filtered_tokens.append(tokens)

    return filtered_tokens

def run_lda(filtered_tokens):
    dictionary = Dictionary(filtered_tokens)
    dictionary.filter_extremes(
        no_below=5, 
        no_above=0.5 
        #keep_n=1000
    )
    corpus = [dictionary.doc2bow(doc) for doc in filtered_tokens]
    lda_model = LdaMulticore(
        corpus=corpus, 
        id2word=dictionary, 
        iterations=100, 
        num_topics=16, 
        workers = 2, 
        passes=100
    )

    return lda_model.print_topics(-1)

content = load("lgbtcol.csv")
cleaned = clean(content) 
run_lda(cleaned)

