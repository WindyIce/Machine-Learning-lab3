from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from feature import NPDFeature


def get_feature():
    """
    A function to change image to NPD feature and save in './datasets/feature/'.
    """
    for i in range(500):
        # change face to feature and store in ./datasts/original/face
        face = np.array(
            Image.open('./datasets/original/face/face_' + str(i).zfill(3) + '.jpg').convert('L').resize((24, 24)))
        npdFeature = NPDFeature(face)
        feature_face = npdFeature.extract()
        pickle.dump(feature_face, open('C:/Users/32699/PycharmProjects/face/datasets/feature/face/face_' + str(i).zfill(3) + '.dat', 'wb'), True)

        # change nonface to feature and store in ./datasets/original/nonface
        nonface = np.array(
            Image.open('./datasets/original/nonface/nonface_' + str(i).zfill(3) + '.jpg').convert('L').resize((24, 24)))
        npdFeature = NPDFeature(nonface)
        feature_nonface = npdFeature.extract()
        pickle.dump(feature_nonface, open('C:/Users/32699/PycharmProjects/face/datasets/feature/nonface/nonface_' + str(i).zfill(3) + '.dat', 'wb'), True)
    print("change end!")


def load_data(n_samples=10):
    """
    Input 
    - n_samples: denotes the number of face images.
    Ootput
    - total_data: a narray includes the face images and nonface images.
    - total_label: a narray includes the labels of all images.(face: 1, nonface: -1)
    """
    mask = np.random.choice(500, n_samples)
    face = []
    nonface = []
    for i in mask:
        face.append(pickle.load(open('./datasets/feature/face/face_' + str(i).zfill(3) + '.dat', 'rb')))
        nonface.append(pickle.load(open('./datasets/feature/nonface/nonface_' + str(i).zfill(3) + '.dat', 'rb')))

    # label the face with 1 and label the nonface with -1
    face = np.array(face)
    face_label = np.ones((n_samples, 1))

    nonface = np.array(nonface)
    nonface_label = np.zeros((n_samples, 1))
    nonface_label = nonface_label - 1

    total_data = np.vstack((face, nonface))
    total_label = np.vstack((face_label, nonface_label))
    return total_data, total_label


if __name__ == "__main__":
    get_feature()
    # get data.
    X, y = load_data(100)
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, random_state=42)

    # train and predict .
    acc_history = []
    number_clf = []
    for i in range(20):
        print(i + 1)
        clf = AdaBoostClassifier(weak_classifier=DecisionTreeClassifier, n_weakers_limit=i + 1)
        clf.fit(X_train, y_train)
        # calculate the accuracy.
        y_pred = clf.predict(X_vali, 0)
        acc = np.mean(y_vali == y_pred)
        acc_history.append(acc)
        number_clf.append(i + 1)

    # the effect of the number of weak classifier.
    plt.title('accuracy history')
    plt.ylabel('accuracy')
    plt.xlabel('number of weak classifier')
    plt.plot(number_clf, acc_history)
    plt.grid()
    plt.show()
