import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import librosa

# Step 1: Initialize paths and data structures
data_path = "Small Data"
categories = ['Faces', 'Fingers', 'Gestures', 'Iris', 'Voices']
subjects = ['A', 'B', 'C']
features_dict = {category: [] for category in categories}
labels_dict = {category: [] for category in categories}

print("Pipeline started. Loading data...")

# Step 2: Define feature extraction functions
def extract_image_features(image_path):
    try:
        print(f"Extracting features from image: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Unable to load image {image_path}. Skipping...")
            return None
        img = cv2.resize(img, (128, 128))  # Resize to manageable size
        hog = cv2.HOGDescriptor()
        features = hog.compute(img)
        if features is None:
            print(f"Error: Unable to extract HOG features from {image_path}. Skipping...")
            return None
        return features.flatten()
    except Exception as e:
        print(f"Exception during feature extraction for {image_path}: {e}")
        return None

def extract_audio_features(audio_path):
    try:
        print(f"Extracting features from audio: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=None)  # Load audio with librosa
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCCs
        return mfccs.mean(axis=1)  # Take the mean of each MFCC coefficient
    except Exception as e:
        print(f"Exception during feature extraction for {audio_path}: {e}")
        return None

# Step 3: Load and extract features
for category in categories:
    category_path = os.path.join(data_path, category)
    for subject in subjects:
        subject_path = os.path.join(category_path, subject)
        for file_name in os.listdir(subject_path):
            file_path = os.path.join(subject_path, file_name)
            if category != 'Voices':  # For images
                features = extract_image_features(file_path)
            else:  # For audio
                features = extract_audio_features(file_path)
            if features is not None and len(features) > 0:  # Only add valid features
                features_dict[category].append(features)
                labels_dict[category].append(subject)

print("Feature extraction completed.")

# Step 4: Ensure consistent feature lengths
print("Ensuring consistent feature lengths...")
max_length = max([
    len(features) for category in features_dict.values() 
    for features in category if features is not None and len(features) > 0
])

def pad_or_truncate(features, target_length):
    if len(features) < target_length:
        return np.pad(features, (0, target_length - len(features)), mode='constant')
    elif len(features) > target_length:
        return features[:target_length]
    return features

all_features = []
all_labels = []
for category in categories:
    print(f"Processing category: {category}")
    for features, label in zip(features_dict[category], labels_dict[category]):
        if features is not None and len(features) > 0:
            features = pad_or_truncate(features, max_length)  # Ensure consistent length
            all_features.append(features)
            all_labels.append(label)

# Convert to NumPy arrays
if len(all_features) > 0:
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
else:
    print("Error: No valid features were extracted. Exiting pipeline.")
    exit()

print(f"Total samples after fusion: {len(all_features)}")

# Step 5: Apply PCA to Reduce Dimensionality
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=50)  # Adjust the number of components as needed
all_features = pca.fit_transform(all_features)
print(f"Features reduced to {all_features.shape[1]} dimensions.")

# Step 6: Handle missing values
print("Handling missing values...")
imputer = SimpleImputer(strategy='mean')
all_features = imputer.fit_transform(all_features)
print("Missing values imputed.")

# Step 7: Normalize features
print("Normalizing features...")
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)
print("Features normalized.")

# Step 8: Train-Test Split
print("Splitting data into train and test sets...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3, random_state=42)
print("Data split completed.")

# Step 9: Train SVM Classifier
print("Training SVM classifier...")
classifier = svm.SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)
print("SVM training completed.")

# Step 10: Evaluate the classifier
print("Evaluating the classifier...")
y_pred = classifier.predict(X_test)
report = classification_report(y_test, y_pred, target_names=subjects)
print("Classification Report:\n")
print(report)
