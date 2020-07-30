import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("pjdata_pre_cg2_nh.csv", names=['Tem', 'Wind', 'Hum', 'Pres',
                                                       'PM10', 'PM2.5', 'O3', 'NO2',
                                                'CO', 'SO2', 'Pat'], header=None)

x1 = df[['Tem', 'Wind', 'Hum', 'Pres', 'PM10', 'PM2.5', 'O3', 'NO2', 'CO', 'SO2']]
y1 = df[['Pat']]
#x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.25, random_state=123456)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=500, test_size=149)

#print(x1_train)
#print(y1_train)

#sns.pairplot(df, hue='Pat')
#plt.show()


# 랜덤 포레스트 하이퍼 파라미터
# n_estimator : 결정 트리의 개수(default 10)
# max_features : 데이터의 feature를 참조할 비율, 개수(auto)
# max_depth : 트리의 깊이
# min_sample_leaf : 리프노드가 되기 위한 최소한의 샘플 데이터 수
# min_samples_split : 노드를 분할하기 위한 최소한의 데이터 수


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


rf = RandomForestClassifier(n_estimators=400, max_features=2, max_depth=10, min_samples_leaf=1, min_samples_split=2, oob_score=True, random_state=3)
print(rf)
rf.fit(x1_train, np.ravel(y1_train))

predicted = rf.predict(x1_test)
accuracy = accuracy_score(y1_test, predicted)

print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
print(f'Mean accuracy score: {accuracy:.3}')



cm = pd.DataFrame(confusion_matrix(y1_test, predicted), columns=[0, 1], index=[0, 1])

sns.heatmap(cm, annot=True, cmap='RdYlGn')
plt.show()

# 중요도
importances = rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
ind = np.argsort(importances)[::-1]

plt.title("IMPORTANCES")
plt.bar(range(x1_test.shape[1]), importances[ind],
        color="b", yerr=std[ind], align="center")
plt.xticks(range(x1_test.shape[1]), ind)
#plt.xlim([-1, x1_test.shape[1]])
plt.show()
'''
# 모델 최적화 (좋은 모델 선택)
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from xgboost import plot_importance
import sklearn

model = XGBClassifier()
rf_param = {
    'n_estimators' : [100, 200, 300, 400],
    'random_state' : [3],
    'max_features' : [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_depth' : [6, 8, 10, 12, 14, 16, 18, 20],
    'min_samples_leaf' : [1, 2, 3, 5, 7],
    'min_samples_split' : [1, 2, 3, 5],
}

rf_grid = GridSearchCV(model, param_grid = rf_param, scoring = "accuracy", n_jobs = -1, verbose = 1)
# n_jobs = -1 -> 모든 CPU 코어를 사용
rf_grid.fit(x1_train, y1_train)
#print(rf_grid.score(x1_test, y1_test))
#pred = rf_grid.predict(x1_test)
#print(sklearn.metrics(y1_test, pred))


print("최고의 평균 정확도 : {0:.4f}".format(rf_grid.best_score_))
print("최고의 파라미터 :   ", rf_grid.best_params_)

# 최고의 평균 정확도 : 0.7180
# 최고의 파라미터 :    {'max_depth': 14, 'min_samples_leaf': 0, 'min_samples_split': 0, 'n_estimators': 300}

cv_result_df = pd.DataFrame(rf_grid.cv_results_)
cv_result_df.sort_values(by=['rank_test_score'], inplace=True)

print(cv_result_df[['params', 'mean_test_score', 'rank_test_score']])


rf_grid = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=1, bootstrap=True, random_state=5)
rf_grid.fit(x1_train, np.ravel(y1_train))
predicted = rf_grid.predict(x1_test)
accuracy = accuracy_score(y1_test, predicted)

cm = pd.DataFrame(confusion_matrix(y1_test, predicted), columns=[0, 1], index=[0, 1])

sns.heatmap(cm, annot=True, cmap='RdYlGn')
plt.show()
#plot_importance(rf_grid)

'''
# 모델 성능 평가 (분류 성능 평가)
from sklearn.metrics import classification_report
print(classification_report(y1_test, predicted, target_names=['0', '1']))

# ROC
import sklearn.metrics as metrics
probs = rf.predict_proba(x1_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y1_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# k-fold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=3)
result = cross_val_score(rf, x1, y1, cv=kfold)
print(result)

result = np.array(result)
print(result.mean())