import pandas as pd

from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

''' reading trained data '''
data = pd.read_csv('train_data.csv')
train_data = data.values[:,:-1]
labels = data.values[:,-1]

test_size = 0.20 
kf = model_selection.KFold(n_splits=10, random_state=42)

x_train, x_test, y_train, y_test = \
model_selection.train_test_split(train_data, labels, test_size=test_size, random_state=42)

model = LinearSVC(C=100.0)
model = DecisionTreeClassifier()
model = RandomForestClassifier()
# model = KNeighborsClassifier()
# model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    # solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    # learning_rate_init=.1)

model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('Accuracy: {:.3f}%'.format(result*100))

results = model_selection.cross_val_score(model, train_data, labels, cv=kf)
# print("Accuracies: {})".format(results))
print("Mean Accuracy: {:.3f}%\nStd Deviation: ({:.3f}%)".format(results.mean()*100, results.std()*100))

# load testing images
test_data = pd.read_csv('test_data.csv')

result_map = { 
			  'name': range(1, len(test_data)+1),
			  'invasive': model.predict(test_data),
			  }
submission_df = pd.DataFrame(result_map)
submission_df.to_csv('submission.csv', index=False, columns=['name', 'invasive'])