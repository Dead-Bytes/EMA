from keras_vggface.vggface import VGGFace
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from tensorflow.keras.layers import Dense, Flatten
import os


def extract_face(img):
    pixels = pyplot.imread(img)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, hieght = results[0]['box']
    x2, y2 = x1 + width, y1 + hieght
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((224,224))
    face_array = asarray(image)
    return face_array

def get_embeddings(filenames):
    yhat = []
    for f in filenames:
        faces = [extract_face(f)]
        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version= 2)
        model = VGGFace(model = 'senet50', include_top=False, input_shape=(224,224,3), pooling='avg')
        yhat.append(model.predict(samples))
    return yhat

def get_embedding(filename):
    face = extract_face(filename)
    sample = asarray(face, 'float32')
    sample = preprocess_input(sample, version= 2)
    model = VGGFace(model = 'senet50', include_top=False, input_shape=(224,224,3), pooling='avg')
    yhat = model.predict(sample)
    return yhat

def is_match(known_embedding, candidate_embedding, thresh = 0.5):
    score = cosine(known_embedding, candidate_embedding)
    if score<= thresh:
        print('>face is a match (%.3f <= %.3f)' % (score, thresh))
    else :
        print('>face is Not a match (%.3f > %.3f)', (score, thresh))

def add_layer(basemodel):
    top_model = basemodel.output
    top_model = Dense(1024, activation ='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(2, activation='softmax')(top_model)
    return top_model

def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,     ### Performing image augmentation to generate differnt images from one image
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.3,
                                     horizontal_flip= True,
                                     fill_mode='nearest')
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=20,
                                                      class_mode='binary',
                                                      target_size=(224, 224))
  validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=20,
                                                                class_mode='binary',
                                                                target_size=(224, 224))
  return train_generator, validation_generator

def read_database():
    embeddings = []
    files = []
    for file in os.listdir('val/'):
        for f in file:
            if f.endswith('.jpg'):
                files.append(f)
                embeddings.append(get_embedding(f))

    return files, embeddings


def train_model(model, training_dir, validation_dir):
    train_generator , validation_generator = train_val_generators(training_dir, validation_dir)
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(train_generator, epochs = 10, validation_data = validation_generator)
    return model

def matcher(filepath, train = False):
    training_dir = 'data/train/'
    validation_dir = 'data/val/'

    model = tf.keras.applications.ResNet50V2(
                        include_top=True,
                        weights='imagenet',
                        classifier_activation='softmax'
                    )
    head = add_layer(model)
    model1 = Model(inputs=model.input, outputs=model.output)
    if train:
        model = train_model(model, training_dir, validation_dir)

    face = extract_face(filepath)
    face_emb = get_embedding(face)
    file_list , emb_list = read_database()
    for i in range(len(emb_list)):
        flag = is_match(face_emb, emb_list[i])
        if flag:
            return file_list[i]
    return 'No Match Found'

if __name__ == '__main__':
    print("initiating_database")
    matcher('val/ben_afflek/httpabsolumentgratuitfreefrimagesbenaffleckjpg.jpg')