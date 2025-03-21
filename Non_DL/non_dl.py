#Necessary imports
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

#Loading all the data and combining it into a single dataframe
#GPT2
file_path_gpt2 = 'AI-Text_Detection/en_gpt2_lines_raw.jsonl'
df_gpt2 = pd.read_json(file_path_gpt2, lines=True)
#GPT3
file_path_gpt3 = 'AI-Text_Detection/en_gpt3_lines_raw.jsonl'
df_gpt3 = pd.read_json(file_path_gpt3, lines=True)
#Human
file_path_human = 'AI-Text_Detection/en_human_lines_raw.jsonl'
df_human = pd.read_json(file_path_human, lines=True)
#llama
file_path_llama = 'AI-Text_Detection/en_llama_lines_raw.jsonl'
df_llama = pd.read_json(file_path_llama, lines=True)

#Combining all the dataframes
df_tot = pd.concat([df_gpt2, df_gpt3, df_human, df_llama])
#Removing a column
df_tot = df_tot[["text", "label"]]

#Performing the train test split
X_train, X_test, y_train, y_test = train_test_split(df_tot['text'], df_tot['label'], test_size=0.2, random_state=0)

#Creating a count vectorizer and transform
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
#Creating a tfidf transformer and transform
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#Creating a RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0)
clf.fit(X_train_tfidf, y_train)

#Transforming the test data
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

#Predicting the labels
y_preds = clf.predict(X_test_tfidf)

#Printing the confusion matrix
labels = ['gpt2', 'gpt3', 'human', 'llama']
cm = confusion_matrix(y_test, y_preds)
print(cm)

#Printing the overall metrics
accuracy = accuracy_score(y_test, y_preds)
precision = precision_score(y_test, y_preds, average='macro')
recall = recall_score(y_test, y_preds, average='macro')
f1 = f1_score(y_test, y_preds, average='macro')

print("Overall Metrics:")
print("Accuracy: ", round(float(accuracy), 4))
print("Precision: ", round(float(precision), 4))
print("Recall: ", round(float(recall), 4))
print("F1 Score: ", round(float(f1), 4))
print()

#Printing the metrics for each class
accuracies_per_class = np.array([np.sum((y_test == i) & (y_preds == i)) / np.sum(y_test == i) for i in np.unique(y_test)])
precision_per_class = precision_score(y_test, y_preds, average=None)
recall_per_class = recall_score(y_test, y_preds, average=None)
f1_per_class = f1_score(y_test, y_preds, average=None)

for i, label in enumerate(sorted(set(y_test))):
    print("Metrics for class ", label)
    print("Accuracy: ", accuracies_per_class[i].round(4))
    print("Precision: ", precision_per_class[i].round(4))
    print("Recall: ", recall_per_class[i].round(4))
    print("F1 Score: ", f1_per_class[i].round(4))
    print()

#Saving the model and the vectorizers
pickle.dump(count_vect, open('count_vect.pkl', 'wb'))
pickle.dump(tfidf_transformer, open('tfidf_transformer.pkl', 'wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))
