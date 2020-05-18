import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

tf.enable_eager_execution()
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

class_names_new = dict()
for key in class_names:
    class_names_new[str(int(key)-1)] = class_names[key]

model_path = 'TrainedModel.h5'

loaded_model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})

def process_image(numpy_image):
    print(numpy_image.shape)
    tensor_img = tf.image.convert_image_dtype(numpy_image, dtype=tf.int16, saturate=False)
    resized_img = tf.image.resize(numpy_image,(IMG_SIZE,IMG_SIZE)).numpy()
    norm_img = resized_img/255

    return norm_img

# TODO: Create the predict function


def predict(image_path, model, top_k):
    
    img = Image.open(image_path)
    test_image = np.asarray(img)

    processed_test_image = process_image(test_image)

    print(processed_test_image.shape, np.expand_dims(processed_test_image,axis=0).shape)
    prob_preds = model.predict(np.expand_dims(processed_test_image,axis=0))
    prob_preds = prob_preds[0].tolist()

#     classes = np.array(sorted(range(len(prob_preds)), key=lambda i: prob_preds[i])[-top_k:])+1
#     classes = sorted(range(len(prob_preds)), key=lambda i: prob_preds[i])[-top_k:]

#     probs = [prob_preds[i] for i in np.argsort(prob_preds)[-top_k:]]
# -------------
#TODO: make these changes in the command line code as well
    values, indices= tf.math.top_k(prob_preds, k=top_k)
#     return values, indices
#     print("jhfdjhdjfh============",values,indices)
#     indices=indices+1
    probs=values.numpy().tolist()#[0]
    classes=indices.numpy().tolist()#[0]

    return probs, classes

probs, classes = predict('./orange_dahlia.jpg',loaded_model, 5)

pred_label_names = [class_names_new[str(idx)] for idx in classes]
print("prediction probabilities :\n",probs)
print('prediction classes:\n',classes)
print('prediction labels:\n',pred_label_names)


img_path_1 = './cautleya_spicata.jpg'
img_path_2 = './hard-leaved_pocket_orchid.jpg'
img_path_3 = './wild_pansy.jpg'
img_path_4 = './orange_dahlia.jpg'

def plot_image(path, index):
#     print('==',path)
    # Plot
    ax = plt.subplot(2, 2, index*2 + 1)
    
    img = Image.open(path)
    test_image = np.asarray(img)
    img = process_image(test_image)
    
    title = path.rsplit("/",1)[-1]
    plt.title(title)
    plt.imshow(img)
    
    # Make prediction
    probs, labels = predict(path,loaded_model, 5)
    print(probs)
    print(labels)
    
    # Get label names
    label_names = [class_names_new[str(idd)] for idd in labels]
    print(label_names)
    
    # Plot bar chart
    ax = plt.subplot(2, 2, index*2 + 2)
    ax.yaxis.tick_right()
    sns.barplot(x=probs, y=label_names, color=sns.color_palette()[0]);

plt.figure(figsize = (6,10))
plot_image(img_path_1, 0)
plot_image(img_path_2, 1)
plt.show()