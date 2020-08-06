import pandas as pd 
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("train.csv")
data = data[['Survived','Pclass','Sex','SibSp','Parch','Fare']]
df = data.drop('Survived', axis='columns')
target = data['Survived']

le_Sex = LabelEncoder()
df['Sex_n'] = le_Sex.fit_transform(df['Sex'])

df_n = df.drop('Sex',axis='columns')

best = 0
for _ in range(30):

	x_train,x_test,y_train,y_test = train_test_split(df_n, target, test_size = 0.15)
	model = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=2)
	model.fit(x_train, y_train)
	acc = model.score(x_test,y_test)
	if acc>best:
		best = acc
		with open("titanic.pickle","wb") as f:
			pickle.dump(model, f)


print(best)