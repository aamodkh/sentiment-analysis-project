import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv("data/training.csv")
positive_string = ""
negative_string = ""
neutral_string = ""

positive = df_train[df_train['class']=="positive"]
negative = df_train[df_train['class']=="negative"]
neutral = df_train[df_train['class']=="neutral"]
for rows in positive['tweet']:
    positive_string+=rows+" "


for rows in negative['tweet']:
    negative_string+=rows+" "

for rows in neutral['tweet']:
    neutral_string+=rows+" "

all_string = ""
for rows in df_train['tweet']:
    all_string+=rows+" "

positive_words = positive_string.split()
negative_words = negative_string.split()
neutral_words = neutral_string.split()
all_words = all_string.split()

positive_frequency = []
positive_probability = []
negative_frequency = []
negative_probability = []
neutral_frequency = []
neutral_probability = []

def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

all_words = Remove(all_words)
#------------------------------------------------------------------------------
for w in all_words:
    positive_frequency.append(positive_words.count(w))

positive_list = pd.DataFrame(
    {'word': all_words, 'frequency': positive_frequency})


# p(w|positive) = (frequency + 1)/(positive_list['frequency'].sum() + positive_list['word'].count())
for x in positive_list['frequency']:
    value = (x+1)/(positive_list['frequency'].sum() + positive_list['word'].count())
    positive_probability.append(value)

positive_list['probability'] = positive_probability

#------------------------------------------------------------------------------

for w in all_words:
    negative_frequency.append(negative_words.count(w))

negative_list = pd.DataFrame(
    {'word': all_words, 'frequency': negative_frequency})


# p(w|negative) = (frequency + 1)/(negative_list['frequency'].sum() + negative_list['word'].count())
for x in negative_list['frequency']:
    value = (x+1)/(negative_list['frequency'].sum() + negative_list['word'].count())
    negative_probability.append(value)

negative_list['probability'] = negative_probability

#------------------------------------------------------------------------------

for w in all_words:
    neutral_frequency.append(neutral_words.count(w))

neutral_list = pd.DataFrame(
    {'word': all_words, 'frequency': neutral_frequency})


# p(w|neutral) = (frequency + 1)/(neutral_list['frequency'].sum() + neutral_list['word'].count())
for x in neutral_list['frequency']:
    value = (x+1)/(neutral_list['frequency'].sum() + neutral_list['word'].count())
    neutral_probability.append(value)

neutral_list['probability'] = neutral_probability

#------------------------------------------------------------------------------

all_list = pd.DataFrame({
    'word':all_words,
    'positive':positive_frequency,
    'positive_probability':positive_probability,
    'negative':negative_frequency,
    'negative_probability':negative_probability,
    'neutral' : neutral_frequency,
    'neutral_probability':neutral_probability
    })
#-------------------------------------------------------------------------------
#Q.N.3
p_positive = positive['class'].count() / df_train['class'].count()
p_negative = negative['class'].count() /df_train['class'].count()
p_neutral =  neutral['class'].count() /df_train['class'].count()


#Q.N.5
df_test = pd.read_csv("data/test.csv")

# positive
final_value = p_positive
ind = []
for text in df_test['tweet']:
    test_tweet = text.split()
    for word in test_tweet:
        prob = positive_list[positive_list['word']==word]
        if prob['probability'].count()==0:
            final_value*=1
        else:
            final_value*=prob['probability'].iloc[0]
    ind.append(final_value)
    final_value = p_positive

df_test['positive_probability'] = ind

#negative
final_value = p_negative
ind = []
for text in df_test['tweet']:
    test_tweet = text.split()
    for word in test_tweet:
        prob = negative_list[negative_list['word']==word]
        if prob['probability'].count()<1:
            final_value*=1
        else:
            final_value*=prob['probability'].iloc[0]
    ind.append(final_value)
    final_value = p_negative

df_test['negative_probability'] = ind

#neutral

final_value = p_negative
ind = []
for text in df_test['tweet']:
    test_tweet = text.split()
    final_value = p_neutral
    for word in test_tweet:
        prob = neutral_list[neutral_list['word']==word]
        if prob['probability'].count()<1:
            final_value*=1
        else:
            final_value*=prob['probability'].iloc[0]
    ind.append(final_value)
    final_value = p_neutral
df_test['neutral_probability'] = ind

#df_test['prediction'] = np.where(df_test['positive_probability']>df_test['negative_probability'],(df_test['positive_probability']>df_test['neutral_probability'],"positive","neutral"),(df_test['negative_probability']>df_test['neutral_probability'],"negative","neutral"))

#-------- classification-----------------#

def compare(a,b,c):
    if(a>b):
        if(a>c):
            return "positive"
        else:
            return "neutral"
    else:
        if(b>c):
            return "negative"
        else:
            return "neutral"

#for row in df_test.iterrows():
# x = compare(row['positive_probability'].iloc[0],row['negative_probability'].iloc[0],row['neutral_probability'].iloc[0])
#  print(row['positive_probability'])

prediction = []
for i in range(len(df_test)) :
    x = compare(df_test.iloc[i, 3], df_test.iloc[i, 4],df_test.iloc[i,5])
    prediction.append(x)

df_test['predicted'] = prediction

df_test.to_csv("data/naive_bayes_results.csv", index=False)

y_test = df_test['class'].tolist()

acc = metrics.accuracy_score(y_test, df_test['predicted'])
acc = acc * 100

print("Accuracy:", acc, "%")
print(metrics.classification_report(y_test, df_test['predicted']))