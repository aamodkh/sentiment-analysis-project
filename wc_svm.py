import os
from PIL import Image
from os import path
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data/svm_results.csv')

text_actual_positive = text_actual_neutral = text_actual_negative = []
text_predicted_positive = text_predicted_neutral = text_predicted_negative = []

for index, row in df.iterrows():
    tweet_id = row['id']
    tweet = row['tweet']
    actual_label_positive = actual_label_neutral = actual_label_negative = row['class']
    predicted_label_positive = predicted_label_neutral = predicted_label_negative = row['prediction']

    if actual_label_positive == "positive":
        text_actual_positive.append(tweet)

    if actual_label_neutral == "neutral":
        text_actual_neutral.append(tweet)

    if actual_label_negative == "negative":
        text_actual_negative.append(tweet)

    if predicted_label_positive == "positive":
        text_predicted_positive.append(tweet)

    if predicted_label_neutral == "neutral":
        text_predicted_neutral.append(tweet)

    if predicted_label_negative == "negative":
        text_predicted_negative.append(tweet)

face_mask = np.array(Image.open("img/plus.png"))
face = np.array(Image.open("img/positive_mask.jpg"))

wordcloud = WordCloud(background_color="white", mask=face_mask, collocations=False).generate(' '.join(text_actual_positive))
image_colors = ImageColorGenerator(face)
wordcloud.recolor(color_func=image_colors)
wordcloud.to_file("img/svm_actual_positive.png")
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

face_mask = np.array(Image.open("img/plus.png"))
face = np.array(Image.open("img/positive_mask.jpg"))

wordcloud = WordCloud(background_color="white", mask=face_mask, collocations=False).generate(' '.join(text_predicted_positive))
image_colors = ImageColorGenerator(face)
wordcloud.recolor(color_func=image_colors)
wordcloud.to_file("img/svm_predicted_positive.png")
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

face_mask = np.array(Image.open("img/neutral_pic.png"))
face = np.array(Image.open("img/neutral_mask.jpg"))

wordcloud = WordCloud(background_color="white", mask=face_mask, collocations=False).generate(' '.join(text_actual_neutral))
image_colors = ImageColorGenerator(face)
wordcloud.recolor(color_func=image_colors)
wordcloud.to_file("img/svm_actual_neutral.png")
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

face_mask = np.array(Image.open("img/neutral_pic.png"))
face = np.array(Image.open("img/neutral_mask.jpg"))

wordcloud = WordCloud(background_color="white", mask=face_mask, collocations=False).generate(' '.join(text_predicted_neutral))
image_colors = ImageColorGenerator(face)
wordcloud.recolor(color_func=image_colors)
wordcloud.to_file("img/svm_predicted_neutral.png")
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

face_mask = np.array(Image.open("img/minus.png"))
face = np.array(Image.open("img/negative_mask.jpg"))

wordcloud = WordCloud(background_color="white", mask=face_mask, collocations=False).generate(' '.join(text_actual_negative))
image_colors = ImageColorGenerator(face)
wordcloud.recolor(color_func=image_colors)
wordcloud.to_file("img/svm_actual_negative.png")
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

face_mask = np.array(Image.open("img/minus.png"))
face = np.array(Image.open("img/negative_mask.jpg"))

wordcloud = WordCloud(background_color="white", mask=face_mask, collocations=False).generate(' '.join(text_predicted_negative))
image_colors = ImageColorGenerator(face)
wordcloud.recolor(color_func=image_colors)
wordcloud.to_file("img/svm_predicted_negative.png")
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
