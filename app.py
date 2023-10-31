import  streamlit as st
import pickle
import sklearn
import  nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import  string
from nltk.corpus import stopwords
stopwords.words('english')
cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Disaster Tweet Classifier')

input_sms = st.text_input('Enter the Tweet')

# 1 . preprocess
if st.button('Predict') :
    def text_trformation(text) :
           # step 1
           text = text.lower()
    # step 2
           words = nltk.word_tokenize(text)
    # step 3
           filtered_words = [i for i in words if i.isalnum()]
    # step 4
           filtered_words = [i for i in filtered_words if i not in stopwords.words('english') and i not in string.punctuation]
    # step 5
           stem_words = [ps.stem(i) for i in filtered_words]
           return ' '.join(stem_words)

    transformed_sms = text_trformation(input_sms)

# 2. vectorize
    vector_input = cv.transform([transformed_sms])

# 3. predict

    result = model.predict(vector_input)[0]

# 4display
    if result == 1 :
           st.header('yes it is a disaster tweet')
    else :
            st.header('not a disaster tweet')
