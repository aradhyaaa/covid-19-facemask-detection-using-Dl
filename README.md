# covid-19-facemask-detection-using-Dl

Mask Detection Project Documentation

Table of Contents
Introduction
Dataset Exploration
Labels Creation
Image Visualization
Data Preprocessing
Data Splitting
Model Creation
Model Training
Model Evaluation
Model Saving
Model Loading and Prediction
Usage
Applications
Future Prospects
Conclusion






1. Introduction
This section introduces the Mask Detection project, outlining the primary goal of detecting whether a person is wearing a mask or not using a convolutional neural network (CNN). The project utilizes a dataset consisting of images with and without masks and employs TensorFlow and Keras for model creation and training.
2. Dataset Exploration
The dataset_exploration section initializes the project by exploring the dataset and obtaining key information.
About the Dataset:
Data set consists of 7553 RGB images in 2 folders as with_mask and without_mask. Images are named as label with_mask and without_mask. Images of faces with mask are 3725 and images of faces without mask are 3828.
Functions Used:
2.1 list_image_files(directory_path)
Lists all image files in a specified directory.
Provides an overview of available images for with and without masks.
2.2 display_sample_images(image_files)
Displays sample images from a list of image files.
Visualizes representative images for better understanding.
3. Labels Creation
The labels_creation section generates labels for images, distinguishing between those with masks (1) and without masks (0).
Functions Used:
3.1 create_labels(num_with_mask, num_without_mask)
Creates labels for images based on the number of images with and without masks.
Establishes ground truth labels for supervised learning.
4. Image Visualization
The image_visualization section displays sample images for both with and without masks.
Functions Used:
4.1 display_image(image_path)
Displays a single image from a specified path.
Allows visual inspection of individual images.
5. Data Preprocessing
The data_preprocessing section prepares the dataset for model training by resizing and converting images to arrays.
Functions Used:
5.1 process_image(image_path, label)
Processes a single image by resizing and converting it to an array.
Converts raw images into a format suitable for machine learning.
5.2 load_and_process_images(image_files, label)
Loads and processes a batch of images with specified labels.
Streamlines the preprocessing of multiple images.
6. Data Splitting
The data_splitting section focuses on dividing the dataset into training and testing sets using scikit-learn's train_test_split.
Functions Used:
6.1 split_dataset(data, labels, test_size, random_state)
Splits the dataset into training and testing sets.
Parameters:
data: The input features or images.
labels: Corresponding labels indicating the class of each input.
test_size: The proportion of the dataset to include in the test split.
random_state: Seed for the random number generator for reproducibility.
Proper data splitting is crucial to assess the model's performance accurately. The test_size parameter controls the size of the test set, ensuring sufficient unseen data for evaluation. Setting random_state ensures reproducibility, making it easier to debug and compare model variations.
7. Model Creation
The model_creation section involves defining a convolutional neural network (CNN) using TensorFlow and Keras.
Functions Used:
7.1 create_mask_detection_model(input_shape, num_classes)
Defines the architecture of the mask detection CNN.
Parameters:
input_shape: The shape of the input data (e.g., (128, 128, 3) for 128x128 RGB images).
num_classes: The number of classes in the classification task (2 in this case for 'with mask' and 'without mask').
The input_shape parameter determines the shape of the input data that the model will expect. It must match the dimensions of the processed images. The num_classes parameter sets the output layer size, aligning with the number of classes in the classification task.
8. Model Training
The model_training section trains the defined model on the training dataset.
Functions Used:
8.1 train_model(model, X_train, Y_train, validation_split, epochs)
Trains the CNN model on the training set with optional validation.
Parameters:
model: The defined neural network model.
X_train: The input features of the training set.
Y_train: The corresponding labels of the training set.
validation_split: The fraction of the training data to be used as validation data during training.
epochs: The number of times the entire training dataset is passed forward and backward through the neural network.
The validation_split parameter aids in monitoring the model's performance on a fraction of the training data not used for training, helping to identify overfitting. Setting the number of epochs controls the duration of training and influences the model's ability to learn from the data.
9. Model Evaluation
The model_evaluation section assesses the performance of the trained model on the testing set.
Functions Used:
9.1 evaluate_model(model, X_test, Y_test)
Evaluates the model on the testing set and prints accuracy.
Parameters:
model: The trained neural network model.
X_test: The input features of the testing set.
Y_test: The corresponding labels of the testing set.
This function provides a quantitative measure of the model's accuracy on unseen data. Understanding the model's performance is critical for assessing its generalization ability to new, unseen images.

10. Model Saving
The model_saving section saves the trained model for future use.
Functions Used:
10.1 save_model(model, file_path)
Saves the trained model to a specified file path.
Parameters:
model: The trained neural network model.
file_path: The path where the model will be saved.
Saving the model allows for later use without the need to retrain. It is a crucial step in preserving the learned knowledge, especially for deploying the model in real-world applications.
11. Model Loading and Prediction
The model_loading_and_prediction section loads the saved model and makes predictions on a sample image.
Functions Used:
11.1 load_and_predict(model_path, input_image_path)
Loads a pre-trained model and makes predictions on an input image.
Parameters:
model_path: The file path where the pre-trained model is saved.
input_image_path: The path of the input image for prediction.
Loading a pre-trained model is crucial for real-world applications where retraining from scratch may not be feasible. Making predictions on new images demonstrates the model's utility in practical scenarios.


12. Usage
This section offers guidance on utilizing the developed code and leveraging the trained model for real-world applications.
Steps to Run the Code:
Ensure that the required libraries, including NumPy, Matplotlib, OpenCV, PIL, scikit-learn, and TensorFlow, are installed.
Adjust the dataset path and file paths within the code according to your local setup.
Run the script to execute the complete workflow, from data exploration to model training and evaluation.
Utilizing the Trained Model:
Once the model is trained and saved (mask_detection_model.keras), you can load it for future use using the provided functions.
To make predictions on new images, use the load_and_predict function, providing the file path of the saved model and the path of the input image.
The model will output predictions.
13. Applications
Public Spaces and Security: Enhance security in public areas by identifying individuals not wearing masks, contributing to public health and safety.
Workplace Safety: Integrate the system into workplaces to ensure employees adhere to safety protocols, minimizing the risk of virus transmission.
Transportation Systems: Deploy in public transportation to monitor and enforce mask-wearing policies, ensuring the safety of passengers and staff.
Retail Environments: Implement in retail spaces for automatic alerting when customers enter without masks, maintaining a secure shopping environment.
14. Future Prospects
Real-Time Monitoring: Integrate real-time monitoring for instantaneous feedback on mask compliance.
Mask Quality Assessment: Enhance the system to assess the quality of masks worn, including coverage of the nose and mouth.
15. Conclusion
In conclusion, this project showcases the development of a Mask Detection system using convolutional neural networks with an accuracy 91.6%.
The trained model holds practical value for applications such as real-time mask detection.

