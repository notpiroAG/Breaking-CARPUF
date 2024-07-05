# Breaking-CARPUF
This repository contains my submission for the assignment in the course CS771: Introduction to Machine Learning under Prof. Purushottam Kar.

## Problem Statement

### Physically Unclonable Functions (PUFs)
A PUF is a function that takes 32 inputs (called challenges) and outputs signals from the upper or lower end of the PUF. If the upper signal arrives first, the output is considered 1; otherwise, it is 0. Mathematically, a PUF can be modeled as a linear machine learning model.

### CAR-PUF
A CAR-PUF uses two arbiter PUFs – a working PUF and a reference PUF – along with a secret threshold value τ > 0. Given a challenge, it is fed into both the working and reference PUFs, and the timings for the upper and lower signals for both PUFs are measured. Let ∆w and ∆r be the differences in timings experienced by the two PUFs for the same challenge. The response to this challenge is 0 if |∆w − ∆r| ≤ τ, and 1 if |∆w − ∆r| > τ, where τ is the secret threshold value.

## Proving the Vulnerabilities of Using a CAR-PUF
The `report.pdf` file contains a detailed explanation of how we can decrypt the CAR-PUF model into a similar linear machine learning model by modifying the original 32 features into a 528 feature-rich model. It includes a comprehensive mathematical proof of the decryption process.

## Training the ML Model
I trained three different linear models and selected the one with the highest accuracy. Below is the code snippet used for selecting the classifiers:
```python
# clf = LinearSVC(C=1.0, max_iter=10000, tol=1e-3, dual=False)
# clf = RidgeClassifier(alpha=1.0, max_iter=10000, tol=1e-5)
clf = LogisticRegression(C=1.0, max_iter=10000, tol=1e-5)
```
## Evaluation metric 
I used average accuracy to evaluate the model's performance:
```python
accuracy  = 0
accuracy += np.average(y_test == pred)
print("Accuracy:", accuracy)
```
## Conclusion
The use of CAR-PUF can be seriously vulnerable to cyber ML attacks as it can be decrypted by a linear machine learning model.
