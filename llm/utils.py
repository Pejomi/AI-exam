import os
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacy.lang.en import English
import langdetect
from wordcloud import WordCloud 
from data_loader import loadWiki, loadFile, loadYoutube
from pytube import Search

def get_data():
    data = []
    # wiki pages
    topics = ['traffic', 'traffic collision', 'traffic sign', 'Road signs in the United Kingdom', 
    'road surface marking', 'Roads in the United Kingdom', 'Accident', 'First aid', 'Driving licence in the United Kingdom', 'Driving under the influence']
    for topic in topics:
        wiki = loadWiki(topic, "en", 1)
        data.extend(wiki)
    # pdf files
    for file in os.listdir("../llm/data/lib/pdf"): 
        file = "../llm/data/lib/pdf/" + file
        data.extend(loadFile(file))
    return data

def save_documents():
    # get data and save to csv
    data = get_data()
    df = pd.DataFrame(data, columns=['page_content', 'metadata', 'type'])
    df['page_content'] = df['page_content'].astype(str)
    df['metadata'] = df['metadata'].astype(str)
    df['type'] = df['type'].astype(str)
    save_documents_to_csv(df)
    # create wordcloud and save image
    image, longstring = wordCloud(df, 'page_content')
    save_word_cloud_image(image)

def load_documents():
    df = pd.read_csv('data/documents.csv')
    return df

def save_documents_to_csv(df):
    df.to_csv("data/documents.csv", encoding='utf-8', index=False)

def get_english_stop_words():
    nlp=English()
    stopw = nlp.Defaults.stop_words
    return stopw

def wordCloud(df, col):
    # replace newlines with spaces   
    longstring = [''.join(x) for x in df[col]]
    longstring = str(longstring).replace('\\n',' ')
    longstring = str(longstring).replace('\n',' ')
    longstring = str(longstring).replace(col,' ')
    # get stopwords
    stopw = get_english_stop_words()
    # remove stopwords
    words = [word for word in longstring.split() if word.lower() not in stopw]
    clean_text = " ".join(words)
    # wordcloud settings
    wordcloud = WordCloud(width=1600, height=800, background_color="white", max_words=1500, contour_width=3, contour_color='steelblue')
    # view
    wordcloud.generate(str(clean_text))
    image = wordcloud.to_image()
    return image,longstring

def save_word_cloud_image(image):
    plt.figure(figsize=(20,10), facecolor='k', frameon=False)
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig('data/wordcloud.png', facecolor='k', bbox_inches='tight')