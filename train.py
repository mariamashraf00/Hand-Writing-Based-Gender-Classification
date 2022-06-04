import glob
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.utils import shuffle
import pickle
from helpers import *

x_train = []
y_train = []

for filename in sorted(glob.glob('images/Females/*.jpg')):
    img = cv2.imread(filename)
    x_train.append(img)
    y_train.append(0)
for filename in sorted(glob.glob('images/Males/*.jpg')):
    img = cv2.imread(filename)
    x_train.append(img)
    y_train.append(1)

x_train, y_train = shuffle(x_train, y_train, random_state=25)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

training_features_hog=[]
training_features_lbp=[]
training_features_glcm=[]
training_features_hinge=[]
for i in range(x_train.shape[0]):
    training_features_hog.append([])
    training_features_hog[i],_=findHOG(x_train[i])
    training_features_lbp.append([])
    training_features_lbp[i]=findLBP(x_train[i])
    training_features_glcm.append([])
    training_features_glcm[i]=findGLCM(x_train[i])
    training_features_hinge.append([])
    training_features_hinge[i]=findHINGE(x_train[i])

training_features=[]
for i in range(x_train.shape[0]):
  training_features.append([])
  training_features[i]=[*training_features_hog[i],*training_features_lbp[i],*training_features_glcm[i],*training_features_hinge[i]]

clf = svm.SVC(kernel='linear',probability=True)
clf.fit(training_features, y_train)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(training_features,y_train)
logreg = LogisticRegression(max_iter=300)
logreg.fit(training_features, y_train)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(training_features, y_train)

eclf = VotingClassifier(estimators=[('LR', logreg), ('RF', rfc), ('KNN', knn), ('SVC', clf)],voting='soft', weights=[1,1,1,2])
eclf.fit(training_features, y_train)

pickle.dump(eclf, open("model.sav", 'wb'))