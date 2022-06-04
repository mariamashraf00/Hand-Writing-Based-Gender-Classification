import glob
from sklearn.ensemble import VotingClassifier
import pickle
import time
from dotenv import dotenv_values
from helpers import *

config = dotenv_values(".env")
test_dir=config['TEST_DIR']
out_dir=config['OUT_DIR']
time_file=out_dir+"times.txt"
result_file=out_dir+"results.txt"

x_test=[]
for filename in sorted(glob.glob(test_dir+'*.jpg')):
    img = cv2.imread(filename)
    x_test.append(img)

model=pickle.load(open("model.sav", 'rb'))
f1 = open(time_file, "a")
f2 = open(result_file, "a")

for i in range(len(x_test)):
    img_original = x_test[i]
    start = time.time()
    test_point = extract_features(img_original)
    prediction = model.predict([test_point])[0]
    end = time.time()
    execution_time = end - start
    f1.write(str(round(execution_time,2))+'\n')
    f2.write(str(prediction)+'\n')

f1.close()
f2.close()