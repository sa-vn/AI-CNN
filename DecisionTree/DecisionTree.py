
# Decision Tree supervised

import os
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import graphviz


# Load Images Function to load iamges from Drive and put the in array
def load_images(base_path, classes, image_size):

  # Initialize lists to store images and labels
 images = []
 labels = []
 for class_folder in classes:
    folder_path = os.path.join(base_path, class_folder)
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert('RGB')
        image = image.resize(image_size)
        image_array = np.array(image)
        #print(image_array.shape)
        images.append(image)
        labels.append(class_folder)  # Use the folder name as the label
  

 # Convert images to a numpy array and normalize(# Normalize pixel values to [0, 1])
 images = np.array(images, dtype=np.float32) / 255.0  
 #print(images.shape)

 # Flatten the images (samples x features) to conver array to 1D array
 images_flattened = images.reshape(len(images), -1)


 # Encode class lables to numerical values
 label_encoder = LabelEncoder()
 labels = label_encoder.fit_transform(labels)

 # return an array of features(image pixele)
 # each row of this array present exteracted featurs from one train iamges
 # and labels array pressent and array of class lables
 return images_flattened, labels

# Define paths and classes
base_path = "/content/drive/MyDrive/ColabNotebooks/DataSet/train"  # Update this with the correct path
classes = ["airport_terminal", "market", "movie_theater", "museum", "restaurant"]
image_size = (256, 256)
X_train = []
Y_train = []
# Load images and labels
X_train,Y_train =load_images(base_path,classes,image_size)

# class lable Encoding
le = LabelEncoder()
lable_classes = le.fit_transform(classes)

length = len(X_train)
print (length)
# Train Decision Tree Classifier
dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, Y_train)
#tree.plot_tree(dtc)

# print a nicer tree using graphviz

# Generate feature names useing numerical indices 
num_features = X_train.shape[1]
feature_names = [f'pixel_{i}' for i in range(num_features)]



dot_data = tree.export_graphviz(dtc, out_file=None,
feature_names=feature_names,
class_names=classes,
filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("DecisionTree") # the DecisionTree will save in a pdf file







