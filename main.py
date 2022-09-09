import pandas as pd
import numpy as np
import requests
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import aiohttp
import asyncio

from PIL import Image
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# data
model = pickle.load(
    open(r'classifier.pkl', 'rb'))
tfidf = open(r'vectorizer.pkl', 'rb')
# ----------------------------Getting Title and Text----------------------------
content_list = []
content_df = []
text_list = []
question_content_list = []
true_index_list = []
false_index_list = []

form_data = pd.DataFrame()


def dict_to_df():
    params = ["id", "title"]
    for content_dict in content_list:
        param_list = {}
        for value in params:
            try:
                param_list[value] = content_dict[value]
            except TypeError or KeyError or ValueError:
                param_list[value] = [""]
            #param_list[value] =content_dict[value]
        df = pd.DataFrame(param_list, index=[content_list.index(content_dict)])
        content_df.append(df)

    concat_df = pd.concat(content_df)
    return concat_df


async def session(id):
    url = 'https://api.jotform.com/form/{}/properties?apiKey={apiKey}'
    async with aiohttp.ClientSession() as session:

        for i in range(len(id)):
            jotform_url = url.format(id[i])

            async with session.get(jotform_url) as resp:
                data_json = await resp.json()
                content = data_json['content']
                content_list.append(content)

        form_data = dict_to_df()
        form_data.to_csv("Form_content.csv")
        return form_data


def question_dict_to_df(id):
    i = 0
    params = ["text", "content", "template"]
    for content_dict in question_content_list:
        if (bool(content_dict) == True):
            df_column = pd.DataFrame(content_dict)
            column_names = df_column.columns.tolist()
            str_list = ""
            for column in column_names:
                for value in params:
                    try:
                        str_list += df_column.loc[value, column] + ","
                    except:
                        str_list += ""
            text_list.append(str_list)
            true_index_list.append(i)
        else:
            false_index_list.append(i)
        i = i+1
    df = pd.DataFrame(text_list)
    df = df.loc[df.index.isin(true_index_list)]
    df["id"] = id.loc[id.index.isin(true_index_list)]
    return df


def set_list(myList):
    question_content_list = myList


async def question_session():
    url = "https://api.jotform.com/form/{}/questions?apiKey={apiKey}"
    data = pd.read_csv("Form_content.csv")
    data['id'] = pd.Series(data['id'], dtype="string")
    id = data.loc[:, 'id']  # formId
    async with aiohttp.ClientSession() as session:
        for i in range(len(id)):
            jotform_url = url.format(id[i])
            async with session.get(jotform_url) as resp:
                data_json = await resp.json()
                content = data_json['content']
                question_content_list.append(content)
                set_list(question_content_list)
        df = question_dict_to_df(id)
        # df_to_excel(df) #write to excel
        df.to_csv("Form_data.csv")  # write to csv
        return df


async def form_set(path):
    data = pd.read_csv(path)
    id = data.loc[:, "id"]  # formId
    title = await session(id)
    text = await question_session()
    return title, text


async def one_form(formID):
    data_id = pd.Series(formID)
    title = await session(data_id)
    text = await question_session()
    return title, text
# ----------------------------End of Getting Data----------------------------

# ----------------------------Preprocessing----------------------------


def merge_title_and_text(title, text):
    # .drop('Unnamed: 0', axis=1)
    data = pd.merge(text, title, how='inner', on=['id'])
    data.rename(columns={'0': 'text'}, inplace=True)
    data["total_text"] = data["title"].astype(
        str) + ", " + data.iloc[:, 0].astype(str)
    data = data.drop('title', axis=1)
    data = data[["id", "total_text"]]
    data = data.drop_duplicates().reset_index(drop=True)
    return data


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
sw = stopwords.words('english')


def tokenizer(text):
    return [stemmer.stem(word) for word in text.split()]


def lemmatizer_func(text_list):
    return [lemmatizer.lemmatize(word) for word in text_list]


def remove_freqwords(text_list):
    FREQWORDS = ['e', 'a', 'an', 'can', 'will', 'mail', 'email', 'address', 'number', 'payment', 'submit', 'phone', 'date', 'form',
                 'may', 'us', 'card', 'example', 'name', 'please', 'com', 'yes', 'no', 'one', 'full', 'like', 'as', 'page', 'would', 'per', 'must']
    return [word for word in text_list if word not in FREQWORDS]


def remove_stopwords(text_list):
    return [word for word in text_list if word not in sw]


def clean_text(text):
    s = BeautifulSoup(text).get_text(strip=True)
    s = s.replace('\u200b', '').replace(
        '\n', '').replace('\xa0', '').replace("\\", '')
    return s


def clean_dataframe_for_text(dataframe):
    for index in range(len(dataframe.loc[:, "total_text"])):
        c_text = clean_text(dataframe.loc[index, "total_text"])
        dataframe.loc[index, "total_text"] = c_text
    return dataframe


def punctuation_and_number_deleting(dataframe):
    # initializing punctuations and number string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~01234567890'''

    for index in range(len(dataframe.loc[:, "total_text"])):
        for ele in dataframe.loc[index, "total_text"]:
            if ele in punc:
                dataframe.loc[index, "total_text"] = dataframe.loc[index, "total_text"].replace(
                    ele, "  ")
    return dataframe


def dataframe_processing(dataframe):

    dataframe = clean_dataframe_for_text(dataframe)
    dataframe = punctuation_and_number_deleting(dataframe)

    for index in range(len(dataframe.loc[:, "total_text"])):
        my_list = tokenizer(dataframe.loc[index, "total_text"])
        my_list = lemmatizer_func(my_list)
        my_list = remove_freqwords(my_list)
        my_list = remove_stopwords(my_list)
        #my_set = set(my_list)
        my_string = ' '.join(str(e) for e in my_list)
        dataframe.loc[index, "total_text"] = my_string
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)
    return dataframe

# ----------------------------End of Preprocessing----------------------------

# ----------------------------Prediction----------------------------


def predict(dataframe):
    df = tfidf.transform(dataframe["total_text"]).toarray()
    prediction = model.predict(df)
    return prediction[0]

# ----------------------------End Of Prediction----------------------------


app = FastAPI()


origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def read_root():
    return {"Root": "Page"}


@app.get("/formID/{formID}")
def read_form(formID: str):

    title, text = asyncio.run(one_form([formID]))
    data = merge_title_and_text(title, text)
    X = dataframe_processing(data)
    t = predict(X)
    print(t)
    print(type(t))
    return {'prediction': t.tolist()}


if __name__ == '__main__':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host='127.0.0.1', port=8000)
