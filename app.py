# %%
import os
import re
import nltk
nltk.data.path.append('./nltk_data/')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import uvicorn
import pickle
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#%% Function


#%%
cntizer = pickle.load(open('cntizer.pickle', 'rb'))
tfizer = pickle.load(open('tfizer.pickle', 'rb'))
list_personality = pickle.load(open('list_personality.pickle','rb'))

lemmatiser = WordNetLemmatizer()

# Remove the stop words for speed 
useless_words = stopwords.words("english")

# Remove these from the posts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]

def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
  list_personality = []
  list_posts = []
  len_data = len(data)
  i=0
  
  for row in data.iterrows():
      #Remove and clean comments
      posts = row[1].posts

      #Remove url links 
      temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

      #Remove Non-words - keep only words
      temp = re.sub("[^a-zA-Z]", " ", temp)

      # Remove spaces > 1
      temp = re.sub(' +', ' ', temp).lower()

      #Remove multiple letter repeating words
      temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

      #Remove stop words
      if remove_stop_words:
          temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
      else:
          temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
          
      #Remove MBTI personality words from posts
      if remove_mbti_profiles:
          for t in unique_type_list:
              temp = temp.replace(t,"")

      # transform mbti to binary vector
      type_labelized = translate_personality(row[1].type) #or use lab_encoder.transform([row[1].type])[0]
      list_personality.append(type_labelized)
      # the cleaned data temp is passed here
      list_posts.append(temp)

  # returns the result
  list_posts = np.array(list_posts)
  list_personality = np.array(list_personality)
  return list_posts, list_personality

b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

def translate_personality(personality):
    # transform mbti to binary vector
    return [b_Pers[l] for l in personality]

#To show result output for personality prediction
def translate_back(personality):
    # transform binary vector to mbti personality
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s
personality_type = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)", 
                   "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"  ]

filename = 'models_personality/models_personality_'
models = []
for l in range(len(personality_type)):
    Y = list_personality[:,l]
    # make predictions for my  data
    model = pickle.load(open(filename + str(l) +'.pickle', 'rb'))
    models.append(model)

# %%
app = FastAPI()

class predictBody(BaseModel):
    message: str


@app.post('/predict')
def predict_personality(predictBody: predictBody):
    my_posts = predictBody.message
    # The type is just a dummy so that the data prep function can be reused
    mydata = pd.DataFrame(data={'type': [''], 'posts': [my_posts]})
    my_posts, dummy = pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)
    my_X_cnt = cntizer.transform(my_posts)
    my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()
    result=[]
    for l in range(len(personality_type)):
        Y = list_personality[:,l]
        # make predictions for my  data
        prediction = models[l].predict(my_X_tfidf)
        result.append(prediction[0])
    print(translate_back(result))
   
    return JSONResponse(content=jsonable_encoder({"result": translate_back(result)}))


@app.get('')
def get_home():
    return {'message': 'Wellcome'}

#%%
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get("PORT", 5000)))

# %%
