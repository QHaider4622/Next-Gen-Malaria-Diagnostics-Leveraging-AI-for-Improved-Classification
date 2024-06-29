# Next-Gen-Malaria-Diagnostics-Leveraging-AI-for-Improved-Classification

This project focuses on classifying images of cells to determine whether they are infected with malaria. The dataset contains cell images labeled as either parasitized or uninfected. We utilize deep learning techniques to build and evaluate models for this classification task. The dataset used in this project is from [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

## Dataset

The dataset consists of images of cells labeled as parasitized or uninfected. The images are used to train and evaluate the deep learning models. You can find more details about the dataset [here](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

## Project Structure

1. **Data Loading and Preprocessing**
    - Loading the dataset
    - Preprocessing the images
    - Data augmentation
      
2. **Model Building**
    - Splitting the data into training and testing sets
    - Defining the deep learning model architecture
    - Compiling the model
    - Training the model
    - Hyperparameter tuning

3. **Model Evaluation**
    - Evaluating model performance
    - Visualizing results
    - Performance comparison of different models
    
## Data Loading and Preprocessing

### Data Loading and Preprocessing

Here, we load the dataset and inspect it 

#### Visulizations of Parasitized or Uninfected

![Visuilizing](https://github.com/QHaider4622/Next-Gen-Malaria-Diagnostics-Leveraging-AI-for-Improved-Classification/assets/79516393/89f4d37c-7740-4a00-914d-96d256ac51f8)

#### Distribution of Labels in Dataset

![Balance](https://github.com/QHaider4622/Next-Gen-Malaria-Diagnostics-Leveraging-AI-for-Improved-Classification/assets/79516393/a066e458-63e2-40d5-9e26-e55477fdad57)

we can see that the dataset is balanced

### Splitting the Dataset into Training, Validation and Testing Sets

The dataset is split into training,validation and testing sets to evaluate the model performance effectively.

### Preprocessing the Images

The images are resized and normalized to prepare them for model training.

### Data Augmentation

Data augmentation techniques are applied to increase the diversity of the training data.

## Model Building

### Defining the Model Architecture

A deep learning model is defined using convolutional neural networks (CNNs).

### Models Used

We experimented with the following models:
- **Custom CNN**
- **Alexnet**

### Compiling the Model

The model is compiled using an appropriate loss function and optimizer

### Training the Model

Both model are trained on the training data from sctrach

## Model Evaluation 

### Evaluating Model Performance

The model are evaluated using metrics such as accuracy, precision, recall, and F1 score.

#### Result of Custom Model on Test Data

#### Result of Alexnet on Test Data

### Visualizing Results

Visualizations are created to understand the model's performance.

#### Performance of Custom Model

#### Performance of Alex Model

### Best Performing Model


The best performing model was ** ** with an accuracy of ** ** and an F1 Score of ** **.

## References
- [Kaggle Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
