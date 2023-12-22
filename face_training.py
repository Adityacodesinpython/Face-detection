import cv2
import os
import numpy

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'..\OpenCV_CSI\Faces\train'


haar_cascade = cv2.CascadeClassifier('haar_face.xml')
# Essentially read in those 33,000 lines of XML code and store that in a variable
# So, the two main classifiers that exist today
# are haar cascades, and mo advanced classifiers core local binary patterns, we're not going
# to talk about local binary patterns at all. But essentially the most advanced
# haar cascade classifiers, they're not as prone to noise in an image as compared to the hard cascades

features = []
labels = []


def create_train_of_images():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            actual_image = cv2.imread(image_path)

            if actual_image is None:
                continue

            gray = cv2.cvtColor(actual_image, cv2.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            # will essentially take this image, use these
            # variables called scale factor and minimum labels to essentially detect a face and return
            # essentially the rectangular coordinates of that face as a list

            for (x,y,w,h) in faces_rect:
                faces_region = gray[y:y+h, x:x+w]
                features.append(faces_region)
                labels.append(label)



create_train_of_images()
print('Training over...')


features = numpy.array(features, dtype='object')
labels = numpy.array(labels)

# And this will essentially instantiate  the face right now
# if version does not match try this:  pip install opencv-contrib-python
# https://stackoverflow.com/questions/45655699/attributeerror-module-cv2-face-has-no-attribute-createlbphfacerecognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Now we can actually train the recognizer on, on the features list, and the labels and the labels list
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')

numpy.save('features.npy', features)
numpy.save('labels.npy', labels)








