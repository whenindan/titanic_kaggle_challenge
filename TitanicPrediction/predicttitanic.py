import pickle
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

pickle_in = open("titanic.pickle","rb")
model = pickle.load(pickle_in)

data = pd.read_csv("test.csv")
inputs = data[['Pclass','Sex','SibSp','Parch','Fare']]
le_Sex = LabelEncoder()
inputs['Sex_n'] = le_Sex.fit_transform(inputs['Sex'])
inputs = inputs.drop('Sex', axis='columns')

predictions = model.predict(inputs)

output = pd.DataFrame({'PassengerId': data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")