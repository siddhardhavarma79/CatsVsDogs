import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset Path
DATASET_PATH = r"E:\CatsVsDogs\cat-and-dog\training_set\training_set"

# Settings
IMG_SIZE = 32
MAX_IMAGES_PER_CLASS = 1000

X = []
y = []

print("Loading images...")

for category in ["cats", "dogs"]:

    folder = os.path.join(DATASET_PATH, category)

    label = 0 if category == "cats" else 1

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    print(f"Reading {category} images...")

    count = 0

    for image_name in os.listdir(folder):

        if count >= MAX_IMAGES_PER_CLASS:
            break

        image_path = os.path.join(folder, image_name)

        try:
            img = cv2.imread(image_path)

            if img is None:
                continue

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Flatten image
            img = img.flatten()

            X.append(img)
            y.append(label)

            count += 1

        except Exception:
            continue

    print(f"Loaded {count} {category} images")

X = np.array(X)
y = np.array(y)

print("\nTotal Images Loaded:", len(X))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training SVM Model...")

# Faster Linear SVM
svm = SVC(
    kernel="linear",
    max_iter=1000
)

svm.fit(X_train, y_train)

print("Training Complete!")

# Prediction
print("Making Predictions...")

y_pred = svm.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Sample Prediction
sample_cat = os.path.join(
    DATASET_PATH,
    "cats",
    os.listdir(os.path.join(DATASET_PATH, "cats"))[0]
)

img = cv2.imread(sample_cat)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.flatten().reshape(1, -1)

prediction = svm.predict(img)

print("\nSample Image Prediction:")
print("Cat" if prediction[0] == 0 else "Dog")