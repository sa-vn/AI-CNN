  """
      The script performs hyperparameter optimization using GridSearchCV to find the best parameters for the decision tree.
      Classes and Methods:
      1. MyDecisionTree: A custom decision tree classifier class with the following methods:
          - load_images: Loads images from specified directories, resizes them, and converts them to numpy arrays.
          - plot_tree: Plots and saves a visual representation of the decision tree using graphviz.
          - decisiontree_evaluate: Evaluates and prints various performance metrics for the classifier.
          - decisiontree_optimization: Optimizes decision tree hyperparameters using GridSearchCV and logs performance metrics.
          - plot_performance: Plots performance improvement per hyperparameter combination.

      Main Script Workflow:
      1. Define the base path to the dataset and the class names.
      2. Create an instance of the MyDecisionTree class.
      3. Load training and validation images and labels.
      4. Perform hyperparameter optimization for min_samples_split, min_samples_leaf, and max_depth.
      5. Evaluate the optimized models on the validation set and plot performance metrics.
  """
  from typing_extensions import Self
  import os
  import numpy as np
  from sklearn import tree
  from sklearn.preprocessing import LabelEncoder
  from PIL import Image
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
  from sklearn.model_selection import GridSearchCV
  import matplotlib.pyplot as plt
  import graphviz

  class MyDecisionTree:

      def load_images(self, base_path, classes, image_size=(256, 256)):
          """
            Load images from specified folders, resize them, and convert to numpy arrays.

            Args:
                base_path (str): Path to the base directory containing class folders.
                classes (list): List of class names corresponding to folder names.
                image_size (tuple): Desired size for resizing images.

            Returns:
                tuple: Flattened images array and corresponding labels.
          """
          # Initialize lists to store images and labels
          images = []
          labels = []
          for class_folder in classes:
              print(class_folder)
              folder_path = os.path.join(base_path, class_folder)
              image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

              for image_file in image_files:
                  image_path = os.path.join(folder_path, image_file)
                  image = Image.open(image_path).convert('RGB')
                  image = image.resize(image_size)
                  image_array = np.array(image)
                  # print(image_array.shape)
                  images.append(image_array)
                  labels.append(class_folder)  # Use the folder name as the label

          # Convert images to a numpy array and normalize(# Normalize pixel values to [0, 1])
          images = np.array(images, dtype=np.float32) / 255.0
          # print(images.shape)

          # Flatten the images (samples x features) to convert array to 1D array
          images_flattened = images.reshape(len(images), -1)

          # Encode class labels to numerical values
          label_encoder = LabelEncoder()
          labels = label_encoder.fit_transform(labels)

          # return an array of features(image pixel)
          # each row of this array present extracted features from one train images
          # and labels array present an array of class labels
          return images_flattened, labels

      def plot_tree(self, dtc, feature_names,classes):
          
          """
            Plot and save a decision tree using graphviz.

            Args:
                dtc (DecisionTreeClassifier): Trained decision tree classifier.
                feature_names (list): List of feature names (optional).
                classes (list): List of class names.
          """
          # print a nicer tree using graphviz
         
          dot_data = tree.export_graphviz(dtc, out_file=None,
                                          feature_names=feature_names,
                                          class_names=classes,
                                          filled=True, rounded=True)
          graph = graphviz.Source(dot_data)
          graph.render("DecisionTree")  # the DecisionTree will save in a pdf file

      def decisiontree_evaluate(self, y_true, y_pred):
          """
                Evaluate and print various performance metrics for a classifier.

                Args:
                    y_true (array): True labels.
                    y_pred (array): Predicted labels.
          """

          accuracy = accuracy_score(y_true, y_pred)
          precision = precision_score(y_true, y_pred, average='macro')
          recall = recall_score(y_true, y_pred, average='macro')
          f1 = f1_score(y_true, y_pred, average='macro')
          conf_matrix = confusion_matrix(y_true, y_pred)
          class_report = classification_report(y_true, y_pred)

          # Print the metrics
          print("Accuracy:", accuracy)
          print("Precision:", precision)
          print("Recall:", recall)
          print("F1 Score:", f1)
          print("Confusion Matrix:\n", conf_matrix)
          print("Classification Report:\n", class_report)




      def decisiontree_optimization(self, X_train, Y_train, X_val, Y_val, param_grid):
        """
        Optimize decision tree hyperparameters using GridSearchCV and log performance metrics.

        Args:
            X_train (array): Training features.
            Y_train (array): Training labels.
            X_val (array): Validation features.
            Y_val (array): Validation labels.
            param_grid (dict): Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.

        Returns:
            tuple: Best model after grid search, performance log.
        """
        dtc = tree.DecisionTreeClassifier(criterion="entropy")
        grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5, return_train_score=True)
        grid_search.fit(X_train, Y_train)

        print("Best hyperparameters:", grid_search.best_params_)
        best_model = grid_search.best_estimator_

        # Log performance metrics
        results = grid_search.cv_results_
        performance_log = {
            'params': results['params'],
            'mean_train_score': results['mean_train_score'],
            'mean_test_score': results['mean_test_score']
        }

        return best_model, performance_log

      def plot_performance(self, performance_log):

       
        """
        Plot performance improvement per hyperparameter combination.

        :param performance_log: Dictionary containing parameters and their corresponding mean train and test scores.
        """
        params = performance_log['params']
        mean_train_score = performance_log['mean_train_score']
        mean_test_score = performance_log['mean_test_score']

        # Extract hyperparameters names and values for plotting
        param_names = list(params[0].keys())
        for param_name in param_names:
            plt.figure(figsize=(10, 6))
            param_values = [param[param_name] for param in params]
            plt.plot(param_values, mean_train_score, label='Train Score', marker='o')
            plt.plot(param_values, mean_test_score, label='Val Score', marker='o')
            plt.xlabel(param_name)
            plt.ylabel('Score')
            plt.title(f'Performance vs {param_name}')
            plt.legend()
            plt.grid(True)
            plt.show()


  if __name__ == "__main__":

    # Update this with the correct path
    base_path = "/content/drive/MyDrive/ColabNotebooks/DataSet/"
    classes = ["airport_terminal", "market", "movie_theater", "museum", "restaurant"]
    image_size = (256, 256)
    X_train = []
    Y_train = []
    odt = MyDecisionTree()

    X_train, Y_train = odt.load_images(os.path.join(base_path, "train"), classes)
    # Load  Val images and labels
    X_val = []
    Y_val = []
    X_val,Y_val =odt.load_images(os.path.join(base_path, "val"),classes)


    print("run decisiontree_optimization function to find best accuracy among 'min_samples_split': [15,20,30]")
    # Hyperparameter optimization min_samples_split
    best_model, performance_log = odt.decisiontree_optimization(X_train, Y_train, X_val, Y_val, {'min_samples_split': [15,20,30]})

    # Evaluate the optimized model on the validation set
    y_val_pred_best=[]
    y_val_pred_best = best_model.predict(X_val)
    print("Optimized Model Evaluation on Validation Set:")
    odt.decisiontree_evaluate(Y_val, y_val_pred_best)

    print("Plot plot_performance")
    # Plot performance comparison
    odt.plot_performance(performance_log)

    print("run decisiontree_optimization function to find best accuracy among 'min_samples_leaf': [15,20,30]")
    # Hyperparameter optimization min_samples_leaf
    best_model, performance_log = odt.decisiontree_optimization(X_train, Y_train, X_val, Y_val, {'min_samples_leaf': [15,20,30]})

    # Evaluate the optimized model on the validation set
    y_val_pred_best=[]
    y_val_pred_best = best_model.predict(X_val)
    print("Optimized Model Evaluation on Validation Set:")
    odt.decisiontree_evaluate(Y_val, y_val_pred_best)

    print("Plot plot_performance")
    # Plot performance comparison
    odt.plot_performance(performance_log)

    print("run decisiontree_optimization function to find best accuracy among 'max_Depth': [15,20,30]}")
    # Hyperparameter optimization
    best_model, performance_log = odt.decisiontree_optimization(X_train, Y_train, X_val, Y_val, {'max_Depth': [15,20,30]})

    # Evaluate the optimized model on the validation set
    y_val_pred_best=[]
    y_val_pred_best = best_model.predict(X_val)
    print("Optimized Model Evaluation on Validation Set:")
    odt.decisiontree_evaluate(Y_val, y_val_pred_best)

    print("Plot plot_performance")
    # Plot performance comparison
    odt.plot_performance(performance_log)