from model import load
from PIL import Image, ImageOps  # Install pillow instead of PIL
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Prepare data
model, class_names = load()
images = [
    # BARN OWL
    { "path": "images/BARN OWL/120.jpg", "true_label": "Barn owl" },
    { "path": "images/BARN OWL/121.jpg", "true_label": "Barn owl" },
    # AMERICAN GOLDFINCH
    { "path": "images/AMERICAN GOLDFINCH/120.jpg", "true_label": "American goldfinch" },
    { "path": "images/AMERICAN GOLDFINCH/121.jpg", "true_label": "American goldfinch" },
    # CARMINE BEE-EATER
    { "path": "images/CARMINE BEE-EATER/120.jpg", "true_label": "Carmine bee-eater" },
    { "path": "images/CARMINE BEE-EATER/121.jpg", "true_label": "Carmine bee-eater" },
    # DOWNY WOODPECKER
    { "path": "images/DOWNY WOODPECKER/120.jpg", "true_label": "Downy woodpecker" },
    { "path": "images/DOWNY WOODPECKER/121.jpg", "true_label": "Downy woodpecker" },
    # EMPEROR PENGUIN
    { "path": "images/EMPEROR PENGUIN/120.jpg", "true_label": "Emperor penguin" },
    { "path": "images/EMPEROR PENGUIN/121.jpg", "true_label": "Emperor penguin" },
    # FLAMINGO
    { "path": "images/FLAMINGO/120.jpg", "true_label": "Flamingo" },
    { "path": "images/FLAMINGO/121.jpg", "true_label": "Flamingo" },
]
 
 # Prepare lists to store the true labels and predicted labels
true_labels = []
predicted_labels = []
results = []

# Iterate through the test data
for test_case in images:
    image_path = test_case["path"]
    true_label = test_case["true_label"]
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Predict with the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    predicted_class = class_names[index]
    confidence_score = prediction[0][index]
    
    # Store the result
    is_correct = (predicted_class.lower().__contains__(true_label.lower()))

    true_labels.append(true_label)
    predicted_labels.append(predicted_class)
    results.append({
        "Image Path": image_path,
        "True Label": true_label,
        "Predicted Class": predicted_class,
        "Confidence Score": confidence_score,
        "Correct Prediction": is_correct
    })

def accuracy_table():
    df = pd.DataFrame(results)
    # Calculate accuracy
    accuracy = df["Correct Prediction"].mean()
    print("Accuracy Table:")
    print(df)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

def confusion_matrix_table():
    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    # Create a dataframe for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def calculate_metrics():
    # Calculate Precision, Recall, and F1-Score for each class
    precision = precision_score(true_labels, predicted_labels, average=None, labels=class_names)
    recall = recall_score(true_labels, predicted_labels, average=None, labels=class_names)
    f1 = f1_score(true_labels, predicted_labels, average=None, labels=class_names)

    # Create a dataframe to summarize the scores
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    # Display the metrics
    print(metrics_df)

accuracy_table()