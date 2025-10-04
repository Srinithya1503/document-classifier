# Python Machine Learning Fundamentals
## Complete Learning Guide for Document Classification Project

---

## Table of Contents
1. [Python Basics](#1-python-basics)
2. [Data Structures](#2-data-structures)
3. [String Operations](#3-string-operations)
4. [Functions](#4-functions)
5. [Imports and Modules](#5-imports-and-modules)
6. [Error Handling](#6-error-handling)
7. [File Operations](#7-file-operations)
8. [NumPy Basics](#8-numpy-basics)
9. [Scikit-learn Core Concepts](#9-scikit-learn-core-concepts)
10. [Machine Learning Pipeline](#10-machine-learning-pipeline)
11. [Model Evaluation](#11-model-evaluation)
12. [Model Persistence](#12-model-persistence)

---

## 1. Python Basics

### Variables and Assignment
```python
# Variables store data
x = 10                    # Integer
name = "John"             # String
is_valid = True           # Boolean
price = 19.99             # Float

# Multiple assignment
a, b, c = 1, 2, 3

# Variable naming conventions
user_name = "Alice"       # snake_case (preferred in Python)
MAX_SIZE = 100            # UPPERCASE for constants
```

### Comments
```python
# Single-line comment

"""
Multi-line comment
or docstring for documentation
"""

'''
Alternative multi-line
comment style
'''
```

### Print Function
```python
# Basic printing
print("Hello, World!")

# Multiple arguments
print("Name:", name, "Age:", 25)

# Formatted strings (f-strings) - Python 3.6+
age = 30
print(f"I am {age} years old")
print(f"Next year: {age + 1}")

# Format specifiers
accuracy = 0.8567
print(f"Accuracy: {accuracy:.2%}")  # Output: Accuracy: 85.67%
print(f"Value: {accuracy:.2f}")     # Output: Value: 0.86
```

---

## 2. Data Structures

### Lists
```python
# Creating lists
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", True, 3.14]

# Accessing elements (0-indexed)
first = fruits[0]         # "apple"
last = fruits[-1]         # "cherry" (negative indexing)

# List methods
fruits.append("orange")   # Add to end
fruits.insert(1, "mango") # Insert at position
fruits.remove("banana")   # Remove specific item
length = len(fruits)      # Get length

# List comprehension
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]

# Slicing
subset = numbers[1:4]     # [2, 3, 4] (start:stop)
```

### Tuples (Immutable lists)
```python
# Creating tuples
coordinates = (10, 20)
single = (42,)            # Note the comma for single element

# Unpacking
x, y = coordinates        # x=10, y=20
```

### Dictionaries (Key-Value pairs)
```python
# Creating dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Accessing values
name = person["name"]              # "Alice"
age = person.get("age", 0)         # 0 is default if key missing

# Adding/modifying
person["email"] = "alice@email.com"
person["age"] = 31

# Iterating
for key, value in person.items():
    print(f"{key}: {value}")

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
```

### Sets (Unique elements)
```python
# Creating sets
unique_numbers = {1, 2, 3, 4, 5}
labels_set = set(["HR", "IT", "HR", "Finance"])  # Duplicates removed

# Set operations
unique_numbers.add(6)
unique_numbers.remove(1)
count = len(unique_numbers)
```

---

## 3. String Operations

### Basic String Methods
```python
text = "Hello World"

# Case conversion
lower = text.lower()           # "hello world"
upper = text.upper()           # "HELLO WORLD"
title = text.title()           # "Hello World"

# Checking content
text.startswith("Hello")       # True
text.endswith("World")         # True
"World" in text                # True

# Splitting and joining
words = text.split()           # ["Hello", "World"]
joined = "-".join(words)       # "Hello-World"

# Stripping whitespace
messy = "  text  "
clean = messy.strip()          # "text"

# String formatting
name = "Alice"
age = 30
message = f"My name is {name} and I am {age} years old"
```

### String Indexing and Slicing
```python
text = "Python"

# Indexing
first = text[0]           # "P"
last = text[-1]           # "n"

# Slicing
sub = text[0:3]           # "Pyt" (start:stop)
sub = text[:3]            # "Pyt" (from beginning)
sub = text[3:]            # "hon" (to end)
reversed_text = text[::-1]  # "nohtyP" (reverse)
```

---

## 4. Functions

### Defining Functions
```python
# Basic function
def greet(name):
    """Function to greet a person"""
    return f"Hello, {name}!"

# Calling the function
message = greet("Alice")

# Function with default parameters
def power(base, exponent=2):
    return base ** exponent

result1 = power(5)        # 25 (uses default exponent=2)
result2 = power(5, 3)     # 125

# Function with multiple return values
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)

minimum, maximum, total = get_stats([1, 2, 3, 4, 5])

# Function with type hints (Python 3.5+)
def add(x: int, y: int) -> int:
    return x + y
```

### Lambda Functions (Anonymous functions)
```python
# Regular function
def square(x):
    return x ** 2

# Lambda equivalent
square_lambda = lambda x: x ** 2

# Common use in sorting
pairs = [(1, 'b'), (2, 'a'), (3, 'c')]
sorted_pairs = sorted(pairs, key=lambda x: x[1])
```

---

## 5. Imports and Modules

### Import Statements
```python
# Import entire module
import math
result = math.sqrt(16)

# Import specific functions
from math import sqrt, pi
result = sqrt(16)

# Import with alias
import numpy as np
array = np.array([1, 2, 3])

# Import everything (not recommended)
from math import *

# Import from submodules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

### Common Imports in Our Project
```python
import joblib                                    # Model persistence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np                               # Numerical operations
```

---

## 6. Error Handling

### Try-Except Blocks
```python
# Basic error handling
try:
    # Code that might raise an error
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Catching multiple exceptions
try:
    file = open("data.txt", "r")
    content = file.read()
except FileNotFoundError:
    print("File not found!")
except IOError:
    print("Error reading file!")
finally:
    # Always executes
    print("Cleanup operations")

# Catching any exception
try:
    risky_operation()
except Exception as e:
    print(f"An error occurred: {e}")

# Example from our project
try:
    pipeline = joblib.load('model_pipeline.joblib')
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    exit(1)
```

---

## 7. File Operations

### Reading Files
```python
# Reading entire file
with open("data.txt", "r") as file:
    content = file.read()

# Reading line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())

# Reading all lines into a list
with open("data.txt", "r") as file:
    lines = file.readlines()
```

### Writing Files
```python
# Writing to a file (overwrites)
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("Second line\n")

# Appending to a file
with open("output.txt", "a") as file:
    file.write("Appended line\n")
```

### Why Use `with` Statement?
```python
# Without 'with' (manual cleanup)
file = open("data.txt", "r")
content = file.read()
file.close()  # Must remember to close!

# With 'with' (automatic cleanup)
with open("data.txt", "r") as file:
    content = file.read()
# File automatically closed, even if error occurs
```

---

## 8. NumPy Basics

### Arrays
```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2], [3, 4]])  # 2D array

# Array properties
print(arr1.shape)      # (5,)
print(arr2.shape)      # (2, 2)
print(arr1.dtype)      # int64

# Array operations
doubled = arr1 * 2     # [2, 4, 6, 8, 10]
added = arr1 + 10      # [11, 12, 13, 14, 15]
```

### Common NumPy Functions
```python
# Statistical functions
arr = np.array([1, 2, 3, 4, 5])
mean = np.mean(arr)           # 3.0
median = np.median(arr)       # 3.0
std = np.std(arr)             # Standard deviation
maximum = np.max(arr)         # 5
minimum = np.min(arr)         # 1

# Finding index of maximum value
max_index = np.argmax(arr)    # 4

# Array slicing
subset = arr[1:4]             # [2, 3, 4]
```

### Used in Our Project
```python
# Getting maximum probability
prediction_probabilities = pipeline.predict_proba(documents)
confidence = np.max(prediction_probabilities) * 100

# Getting index of maximum value
max_index = np.argmax(prediction_probabilities)
```

---

## 9. Scikit-learn Core Concepts

### What is Scikit-learn?
Scikit-learn is a Python library for machine learning that provides:
- Data preprocessing tools
- Machine learning algorithms
- Model evaluation metrics
- Pipeline utilities

### Basic Workflow
```python
# 1. Prepare data
X = ["text1", "text2", "text3"]  # Features (input)
y = ["A", "B", "A"]              # Labels (output)

# 2. Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)      # Training

# 4. Make predictions
predictions = model.predict(X_test)

# 5. Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
```

### Important Parameters

#### `train_test_split`
```python
train_test_split(
    X,                      # Feature data
    y,                      # Labels
    test_size=0.2,         # 20% for testing, 80% for training
    random_state=42,       # Seed for reproducibility
    stratify=y             # Maintain class distribution
)
```

#### `random_state`
```python
# Controls randomness for reproducibility
# Same random_state = same results every time
model = LogisticRegression(random_state=42)

# Different runs with random_state=42 give identical results
# Without random_state, results vary each time
```

---

## 10. Machine Learning Pipeline

### What is a Pipeline?
A Pipeline chains multiple processing steps into a single object:
```python
from sklearn.pipeline import Pipeline

# Without Pipeline (manual steps)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Prediction requires remembering both objects
X_test_vec = vectorizer.transform(X_test)
predictions = model.predict(X_test_vec)

# With Pipeline (automatic chaining)
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Training applies all steps
pipeline.fit(X_train, y_train)

# Prediction automatically vectorizes
predictions = pipeline.predict(X_test)
```

### Pipeline Benefits
1. **Cleaner code**: One object instead of many
2. **Prevent errors**: Can't forget preprocessing steps
3. **Easy deployment**: Save/load entire pipeline
4. **Cross-validation**: Works seamlessly with CV

### TF-IDF Vectorizer Explained
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=100,        # Keep only top 100 words
    ngram_range=(1, 2),      # Use single words and word pairs
    stop_words='english',    # Remove common words (the, is, at)
    lowercase=True           # Convert all text to lowercase
)

# Example transformation
documents = [
    "machine learning is great",
    "deep learning is powerful"
]

# Converts text to numerical matrix
X = vectorizer.fit_transform(documents)
# Result: Matrix where each column represents a word/phrase
```

#### What is TF-IDF?
- **TF** (Term Frequency): How often a word appears in a document
- **IDF** (Inverse Document Frequency): How rare/unique a word is across all documents
- **TF-IDF** = TF × IDF: Words that are common in one document but rare overall get high scores

Example:
```
Document 1: "machine learning is great"
Document 2: "deep learning is powerful"

Word "learning": Appears in both → Lower IDF → Lower TF-IDF
Word "machine": Appears in one → Higher IDF → Higher TF-IDF
Word "is": Common stop word → Removed
```

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(
    max_iter=1000,              # Maximum training iterations
    random_state=42,            # For reproducibility
    solver='lbfgs',             # Optimization algorithm
    multi_class='multinomial'   # Handle multiple classes
)

# Training
classifier.fit(X_train, y_train)

# Prediction
predictions = classifier.predict(X_test)

# Prediction probabilities
probabilities = classifier.predict_proba(X_test)
# Returns: [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], ...]
#           Class1 Class2 Class3 for each document
```

---

## 11. Model Evaluation

### Accuracy Score
```python
from sklearn.metrics import accuracy_score

y_true = ["A", "B", "A", "C"]
y_pred = ["A", "B", "C", "C"]

accuracy = accuracy_score(y_true, y_pred)
# accuracy = 0.75 (3 out of 4 correct)

# As percentage
print(f"Accuracy: {accuracy:.2%}")  # Accuracy: 75.00%
```

### Classification Report
```python
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print(report)

# Output:
#               precision    recall  f1-score   support
#
#           HR       0.85      0.90      0.87         10
#      Finance       0.90      0.85      0.87         10
#           IT       0.88      0.88      0.88         10
#
#     accuracy                           0.88         30
#    macro avg       0.88      0.88      0.88         30
# weighted avg       0.88      0.88      0.88         30
```

#### Understanding Metrics:

**Precision**: Of all items predicted as class X, how many were actually class X?
```
Precision = True Positives / (True Positives + False Positives)
```

**Recall**: Of all actual class X items, how many did we correctly identify?
```
Recall = True Positives / (True Positives + False Negatives)
```

**F1-Score**: Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Support**: Number of actual occurrences of each class

### Prediction Probabilities
```python
# Get class probabilities
probabilities = pipeline.predict_proba(documents)

# For 3 classes (HR, Finance, IT):
# [[0.7, 0.2, 0.1],    # Document 1: 70% HR, 20% Finance, 10% IT
#  [0.1, 0.8, 0.1],    # Document 2: 10% HR, 80% Finance, 10% IT
#  [0.2, 0.1, 0.7]]    # Document 3: 20% HR, 10% Finance, 70% IT

# Get confidence (highest probability)
confidence = np.max(probabilities[0]) * 100  # 70.0%

# Get class names
classes = pipeline.classes_  # ['Finance', 'HR', 'IT']

# Create probability dictionary
prob_dict = dict(zip(classes, probabilities[0]))
# {'Finance': 0.2, 'HR': 0.7, 'IT': 0.1}
```

---

## 12. Model Persistence

### Saving Models with Joblib
```python
import joblib

# Train your model/pipeline
pipeline = Pipeline([...])
pipeline.fit(X_train, y_train)

# Save to disk
joblib.dump(pipeline, 'model_pipeline.joblib')
print("Model saved!")
```

### Loading Models
```python
import joblib

# Load from disk
pipeline = joblib.load('model_pipeline.joblib')

# Use immediately (no retraining needed)
predictions = pipeline.predict(new_documents)
```

### Why Use Joblib?
- **Efficient**: Faster than pickle for large numpy arrays
- **Compressed**: Smaller file sizes
- **Standard**: Industry standard for scikit-learn models

### Alternative: Pickle
```python
import pickle

# Saving
with open('model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

# Loading
with open('model.pkl', 'rb') as file:
    pipeline = pickle.load(file)
```

---

## 13. Complete Example Walkthrough

Let's break down the key code from our project:

### Training Pipeline
```python
# Step 1: Import libraries
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Step 2: Prepare data
documents = ["text1", "text2", "text3", ...]
labels = ["HR", "Finance", "IT", ...]

# Step 3: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    documents,           # Input features
    labels,              # Target labels
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    stratify=labels     # Balanced split
)

# Step 4: Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        stop_words='english'
    )),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Step 5: Train
pipeline.fit(X_train, y_train)

# Step 6: Evaluate
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")

# Step 7: Save
joblib.dump(pipeline, 'model_pipeline.joblib')
```

### Prediction Service
```python
# Step 1: Load model
import joblib
import numpy as np

pipeline = joblib.load('model_pipeline.joblib')

# Step 2: Prepare new data
new_documents = [
    "employee performance review",
    "budget allocation planning"
]

# Step 3: Make predictions
predictions = pipeline.predict(new_documents)
# Output: ['HR', 'Finance']

# Step 4: Get probabilities
probabilities = pipeline.predict_proba(new_documents)
# Output: [[0.8, 0.1, 0.1], [0.1, 0.85, 0.05]]

# Step 5: Extract confidence
for doc, pred, probs in zip(new_documents, predictions, probabilities):
    confidence = np.max(probs) * 100
    print(f"Document: {doc}")
    print(f"Predicted: {pred}")
    print(f"Confidence: {confidence:.2f}%")
```

---

## 14. Common Patterns and Idioms

### List Iteration
```python
# Basic iteration
for item in items:
    print(item)

# With index
for i, item in enumerate(items):
    print(f"{i}: {item}")

# With index starting from 1
for i, item in enumerate(items, 1):
    print(f"{i}. {item}")
```

### Zip Function (Parallel iteration)
```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

# Iterate over multiple lists simultaneously
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# With three lists
docs = ["doc1", "doc2"]
preds = ["HR", "IT"]
probs = [[0.8, 0.2], [0.3, 0.7]]

for doc, pred, prob in zip(docs, preds, probs):
    print(f"{doc}: {pred} ({prob})")
```

### String Repetition
```python
# Repeat character/string
line = "=" * 50        # "=================================================="
spaces = " " * 10      # "          "
```

### F-string Formatting
```python
value = 3.14159

# Decimal places
print(f"{value:.2f}")      # "3.14"

# Percentage
print(f"{value:.1%}")      # "314.2%"

# Padding
print(f"{value:10.2f}")    # "      3.14" (10 chars wide)

# Currency
print(f"${value:.2f}")     # "$3.14"
```

---

## 15. Best Practices

### Code Style
```python
# Good variable names
user_count = 100          # Clear and descriptive
max_iterations = 1000     # Full word, not abbreviation

# Bad variable names
uc = 100                  # Too short
maxIter = 1000            # camelCase (use snake_case)

# Function naming
def calculate_accuracy(predictions, labels):  # Verb + noun
    pass

# Constants
MAX_FEATURES = 100        # UPPERCASE
DEFAULT_RANDOM_STATE = 42
```

### Comments
```python
# Good: Explain WHY, not WHAT
# Use stratify to maintain class balance in small datasets
X_train, X_test = train_test_split(X, y, stratify=y)

# Bad: States the obvious
# Split the data
X_train, X_test = train_test_split(X, y)
```

### Error Messages
```python
# Good: Helpful and actionable
try:
    pipeline = joblib.load('model.joblib')
except FileNotFoundError:
    print("ERROR: Model file not found.")
    print("Please run 'train_pipeline.py' first to train the model.")
    exit(1)

# Bad: Vague
except FileNotFoundError:
    print("Error!")
```

---

## 16. Quick Reference

### Data Types
```python
int        # 42, -10
float      # 3.14, -0.5
str        # "hello", 'world'
bool       # True, False
list       # [1, 2, 3]
tuple      # (1, 2, 3)
dict       # {"key": "value"}
set        # {1, 2, 3}
```

### Operators
```python
+          # Addition
-          # Subtraction
*          # Multiplication
/          # Division (float result)
//         # Integer division
%          # Modulo (remainder)
**         # Exponentiation
==         # Equality
!=         # Inequality
<, >, <=, >=  # Comparisons
and, or, not  # Boolean operators
```

### Common Methods
```python
len(x)              # Length
str(x)              # Convert to string
int(x)              # Convert to integer
float(x)            # Convert to float
type(x)             # Get type
print(x)            # Print to console
range(n)            # Generate sequence 0 to n-1
enumerate(x)        # Add counter to iterable
zip(x, y)           # Combine iterables
```

---

## 17. Practice Exercises

### Exercise 1: Lists and Loops
```python
# Create a list of numbers and print their squares
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    print(f"{num} squared is {num**2}")
```

### Exercise 2: Dictionary Operations
```python
# Create a dictionary and iterate over it
scores = {"Alice": 85, "Bob": 92, "Charlie": 78}
for name, score in scores.items():
    print(f"{name} scored {score}")
```

### Exercise 3: String Manipulation
```python
# Clean and process text
text = "  Machine Learning is AMAZING!  "
cleaned = text.strip().lower()
words = cleaned.split()
print(words)  # ['machine', 'learning', 'is', 'amazing!']
```

### Exercise 4: List Comprehension
```python
# Create list of even numbers
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]
```

---

## 18. Additional Resources

### Official Documentation
- **Python**: https://docs.python.org/3/
- **NumPy**: https://numpy.org/doc/
- **Scikit-learn**: https://scikit-learn.org/stable/
- **Pandas**: https://pandas.pydata.org/docs/

### Learning Paths
1. **Python Basics** → Control flow, data structures, functions
2. **NumPy** → Arrays, mathematical operations
3. **Pandas** → Data manipulation (for real datasets)
4. **Scikit-learn** → Machine learning algorithms
5. **Advanced Topics** → Deep learning, neural networks

### Next Steps
- Experiment with different classifiers (Random Forest, SVM)
- Try hyperparameter tuning with GridSearchCV
- Work with real-world datasets
- Learn about feature engineering
- Explore deep learning with TensorFlow or PyTorch

---

## Glossary

**Algorithm**: Step-by-step procedure for solving a problem

**Classification**: Predicting which category an item belongs to

**Feature**: Input variable used for making predictions

**Label**: Output category or value we want to predict

**Model**: Mathematical representation learned from data

**Pipeline**: Series of data processing steps chained together

**Training**: Process of learning patterns from data

**Prediction**: Using a trained model to classify new data

**Accuracy**: Percentage of correct predictions

**Overfitting**: Model memorizes training data but fails on new data

**Underfitting**: Model is too simple to capture patterns

---

**You now have all the fundamental knowledge needed to understand and extend the document classification project!** 🎉