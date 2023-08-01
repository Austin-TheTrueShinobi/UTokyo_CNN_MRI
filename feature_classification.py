import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def extract_features(mri_data, n_components=10):
    """
    Extract principal components from MRI data using PCA.
    
    Parameters:
    - mri_data: MRI data as a numpy array.
    - n_components: Number of principal components to extract.
    
    Returns:
    - features: Extracted features as a numpy array.
    """
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(mri_data.reshape(mri_data.shape[0], -1))
    return features

def classify_features(features, labels):
    """
    Classify MRI features using a Support Vector Machine (SVM).
    
    Parameters:
    - features: Extracted features as a numpy array.
    - labels: Ground truth labels for the MRI data.
    
    Returns:
    - report: Classification report.
    """
    clf = SVC()
    clf.fit(features, labels)
    predictions = clf.predict(features)
    report = classification_report(labels, predictions)
    return report

# Example usage:
labels = np.array([0, 1, 0, 1, 1])  # Sample labels for the MRI data
features = extract_features(mri_data)
report = classify_features(features, labels)
print(report)
