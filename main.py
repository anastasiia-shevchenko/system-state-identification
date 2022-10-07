import pandas as pd # for data manipulation
import numpy as np # for data manipulation
from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.ensemble import AdaBoostClassifier # for AdaBoost model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import time
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score




def loadSimpData():
    Data_Frame = pd.read_csv("Data.csv",delimiter=';')
    Data_Frame1 = pd.read_csv("Data_Normal.csv",delimiter=',')
    Data_Frame1=Data_Frame1.drop(["S","T","U","X"],axis=1)
    #x_Drop = Data_Frame.drop(["Vir"],axis=1)
    x_Drop = Data_Frame.drop(["U", "V", "W", "X", "Vir"], axis=1)
    y_Drop = Data_Frame["Vir"]
    # # -------Для перевода в бинарность--------------------------------------
    # X['Value'] = X['Value'].astype('category').cat.rename_categories(['0', '1'])
    # encoder = LabelEncoder()
    # binary_encoded_y = pd.Series(encoder.fit_transform(X['Value']))
    # # ----------------------------------------------------------------------
    return Data_Frame, x_Drop, y_Drop.values, y_Drop, Data_Frame1
    
    
    
    
    
    
    
    def plot_trees(trees):
    c=1
    for model in trees:
        tree.plot_tree(model)
        fn = ["A", "B", "C", "D"]
        cn = ['1', '0']
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), dpi=700)
        tree.plot_tree(model,
                       feature_names=fn,
                       class_names=cn,
                       filled=True,
                       node_ids=True,
                       rounded=True,
                       proportion=True)
        fig.savefig(str(c)+'_imagenam.jpg', dpi=600, format='jpg', bbox_inches='tight', pad_inches=0.05)
        c+=1
        
        
        # Ф-ция визуальзации
def show_DataSet(df):
    #sns.pairplot(df, height=10, aspect=1,  vars=df.columns[:-1], hue="Vir")
    sns.pairplot(df, vars=df.columns[:-1], hue="Vir")

    #g.fig.set_size_inches(150, 150)
    plt.show()
    
    
    def Tree_Summary():
    print('*************** Общая Информация ***************')
    print('Количество классов: ', clf.n_classes_)
    print('Метки классов: ', clf.classes_)
    print('Количество классификаторов: ', len(clf.estimators_))
    print('Базовая оценка, из которой вырос ансамбль: ', clf.base_estimator_)
    print('Ошибка классификации(min/max): ', min(clf.estimator_errors_), "/",
          max(clf.estimator_errors_))  # Ошибка классификации для каждой оценки в усиленном ансамбле.
    print('--------------------------------------------------------')
    print("")

    print('*************** Оценка на тестовых данных ***************')
    # Возвращает среднюю точность для заданных тестовых данных и меток.
    score_te = model.score(X_test, y_test)
    print('Оценка точности: ', score_te)
    # Посмотрите отчет о классификации, чтобы оценить модель
    print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')
    print("")

    print('*************** Оценка тренировочных данных ***************')
    score_tr = model.score(X_train, y_train)
    print('Оценка точности: ', score_tr)
    # Посмотрите отчет о классификации, чтобы оценить модель
    print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')

df, X, y, y_no_val, df_1 = loadSimpData()
#Это функция уровня осей, и она будет отображать тепловую карту в активных в данный момент осях, если ни одна из них не указана в axаргументе. 
#Часть этого пространства осей будет занята и использована для построения карты цветов, если не cbar установлено значение False или для cbar_ax.

plt.figure(figsize=(10,10))

corr_matrix = df.corr()
lower = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(np.bool))
plt.figure(figsize=(12,12),dpi=200)
sns.heatmap(lower, annot=True, fmt='.2f', cbar=False, center=0);


show_DataSet(df) # Ф-ция визуальзации


#### разбиение на тестовые и тренировочные выборки при использовании Data_Anomal и Data_Normal
X_train = X
y_train = y_no_val
X_test = df_1
y_test = [0] * 1500 # Для Data_Normal
#y_test = [1] * 1500 # Для Data_Anomal
#### разбиение на тестовые и тренировочные выборки при использовании только Data.csv
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)



# ##### Step 3
# Set model and its parameters
model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=10,  # Количество деревьев
    learning_rate=1.0,  # Bес, применяемый к каждому классификатору на каждой итерации повышения
    algorithm='SAMME.R',# Если "SAMME.R", тогда используйте реальный алгоритм повышения SAMME.R. base_estimatorдолжен поддерживать расчет вероятностей классов. Если «SAMME», тогда используйте алгоритм дискретного повышения SAMME
    random_state=0,     # Управляет случайным начальным значением, заданным base_estimatorна каждой итерации повышения.
)

clf=model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Матрица неточностей
# [[число] - истино-положительное решение   [число] - ложные срабатывания
#  [число] - ложные отрицания               [число] - истино-отрицательное решение
# ]
print(confusion_matrix(y_test, predictions))


##### Step 4
# Predict class labels on training data
pred_labels_tr = model.predict(X_train)
# Predict class labels on a test data
pred_labels_te = model.predict(X_test)
Tree_Summary()
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
def loadSimpData():
    Data_Frame = pd.read_csv("Data.csv",delimiter=';')
    Data_Frame1 = pd.read_csv("Data_Normal.csv",delimiter=',')
    Data_Frame1=Data_Frame1.drop(["S","T","U","X"],axis=1)
    #x_Drop = Data_Frame.drop(["Vir"],axis=1)
    x_Drop = Data_Frame.drop(["U", "V", "W", "X", "Vir"], axis=1)
    y_Drop = Data_Frame["Vir"]
    # # -------Для перевода в бинарность--------------------------------------
    # X['Value'] = X['Value'].astype('category').cat.rename_categories(['0', '1'])
    # encoder = LabelEncoder()
    # binary_encoded_y = pd.Series(encoder.fit_transform(X['Value']))
    # # ----------------------------------------------------------------------
    return Data_Frame, x_Drop, y_Drop.values, y_Drop, Data_Frame1
df, X, y, y_no_val, df_1 = loadSimpData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
probs = model.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)

#ОПРЕДЕЛЕНИЕ ВРЕМЕНИ РАБОТЫ МОДЕЛИ
working_hours_1=[]#max_depth=2
for i in range(50):
    start = time.time()
    model = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=i+1,  # Количество деревьев
        learning_rate=1.0,  # Bес, применяемый к каждому классификатору на каждой итерации повышения
        algorithm='SAMME.R',# Если "SAMME.R", тогда используйте реальный алгоритм повышения SAMME.R. base_estimatorдолжен поддерживать расчет вероятностей классов. Если «SAMME», тогда используйте алгоритм дискретного повышения SAMME
        random_state=0,     # Управляет случайным начальным значением, заданным base_estimatorна каждой итерации повышения.
    )
    clf = model.fit(X_train, y_train)
    end = time.time()
    f=end-start
    working_hours_1.append(f)
    
    working_hours_2=[]#max_depth=3
for i in range(50):
    start = time.time()
    model = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3),
        n_estimators=i+1,  # Количество деревьев
        learning_rate=1.0,  # Bес, применяемый к каждому классификатору на каждой итерации повышения
        algorithm='SAMME.R',# Если "SAMME.R", тогда используйте реальный алгоритм повышения SAMME.R. base_estimatorдолжен поддерживать расчет вероятностей классов. Если «SAMME», тогда используйте алгоритм дискретного повышения SAMME
        random_state=0,     # Управляет случайным начальным значением, заданным base_estimatorна каждой итерации повышения.
    )
    clf = model.fit(X_train, y_train)
    end = time.time()
    f=end-start
    working_hours_2.append(f)
    
    
working_hours_3=[]#max_depth=4
for i in range(50):
    start = time.time()
    model = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=4),
        n_estimators=i+1,  # Количество деревьев
        learning_rate=1.0,  # Bес, применяемый к каждому классификатору на каждой итерации повышения
        algorithm='SAMME.R',# Если "SAMME.R", тогда используйте реальный алгоритм повышения SAMME.R. base_estimatorдолжен поддерживать расчет вероятностей классов. Если «SAMME», тогда используйте алгоритм дискретного повышения SAMME
        random_state=0,     # Управляет случайным начальным значением, заданным base_estimatorна каждой итерации повышения.
    )
    clf = model.fit(X_train, y_train)
    end = time.time()
    f=end-start
    working_hours_3.append(f)
    
    working_hours_4=[]#max_depth=5
for i in range(50):
    start = time.time()
    model = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=5),
        n_estimators=i+1,  # Количество деревьев
        learning_rate=1.0,  # Bес, применяемый к каждому классификатору на каждой итерации повышения
        algorithm='SAMME.R',# Если "SAMME.R", тогда используйте реальный алгоритм повышения SAMME.R. base_estimatorдолжен поддерживать расчет вероятностей классов. Если «SAMME», тогда используйте алгоритм дискретного повышения SAMME
        random_state=0,     # Управляет случайным начальным значением, заданным base_estimatorна каждой итерации повышения.
    )
    clf = model.fit(X_train, y_train)
    end = time.time()
    f=end-start
    working_hours_4.append(f)
    
    
    #График времени обучения в зависимости от количества деревьев и их глубины
    
    
    x = range(1, 51)
plt.figure(figsize=(30, 7))

plt.plot(x, working_hours_1, 'o-r', alpha=0.7, label="max_depth=2", lw=2, mec='b', mew=1, ms=5)
plt.plot(x, working_hours_2, 'o-b', alpha=0.7, label="max_depth=3", lw=2, mec='b', mew=1, ms=5)
plt.plot(x, working_hours_3, 'o-g', alpha=0.7, label="max_depth=4", lw=2, mec='b', mew=1, ms=5)
plt.plot(x, working_hours_4, 'o-k', alpha=0.7, label="max_depth=5", lw=2, mec='b', mew=1, ms=5)
plt.legend()
plt.grid(True)

plt.xlabel('Количество деревьев')
plt.ylabel('Время обучения')
