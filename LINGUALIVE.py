# Installed package: pandas, seaborn, matplotlib, scikit-learn, SpeechRecognition, PyAudio

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import threading

# -------------------- Prepare the Dataset --------------------

# Load the dataset containing text samples and their corresponding languages
data = pd.read_csv('Language-Detection.csv', encoding='utf-8')

# Remove any rows with missing values
data = data.dropna()

# Display the language counts
print("Language counts in the dataset:")
print(data["Language"].value_counts())

# Extract features,x (text) and labels.y (languages)
X = data["Text"]
y = data["Language"]

# -------------------- Train the Language Detection Model --------------------

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Create a machine learning pipeline with TF-IDF vectorization and a Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the pipeline on the training data
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Evaluate the model's accuracy of the test data
accuracy_percentage = round(accuracy_score(y_test, y_pred) * 100, 2)
print(f"Model Accuracy: {accuracy_percentage}%")

confusion_mat = confusion_matrix(y_test, y_pred)
# The confusion matrix plot
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# -------------------- Create the LinguaLive Application --------------------

# Function to capture and recognize speech input
def capture_speech_input():
    recognizer = sr.Recognizer()  # Initialize the speech recognizer
    with sr.Microphone() as microphone_source:
        try:
            # Update the status label to indicate listening
            status_label.config(text="Listening...")

            # Listen to the audio input from the user
            audio_input = recognizer.listen(microphone_source, timeout=5)

            # Convert the audio input to text using Google's speech recognition API
            recognized_text = recognizer.recognize_google(audio_input)
            return recognized_text
        except sr.UnknownValueError:
            # Handle case where the speech cannot be understood
            messagebox.showerror("Error", "Could not understand the audio. Please try again.")
        except sr.RequestError:
            # Handle case where there's an issue with the speech recognition service
            messagebox.showerror("Error", "Error connecting to the speech recognition service. Check your internet.")
        except Exception as e:
            # Handle any other exceptions
            messagebox.showerror("Error", str(e))
        return None


# Function to detect the language of the given text using the trained model
def detect_language_of_text(input_text):
    if input_text:
        # Use the trained language detection model to predict the language
        predicted_language = model.predict([input_text])[0]
        return predicted_language
    return "Unknown"


# Function to process speech input and detect its language
def process_speech_to_language():
    try:
        # Capture the speech input and convert it to text
        speech_text = capture_speech_input()

        if speech_text:
            # Detect the language of the captured text
            detected_language = detect_language_of_text(speech_text)

            # Update the labels to display the detected text and language
            detected_text_label.config(text=f"Recognized Text: {speech_text}")
            detected_language_label.config(text=f"Detected Language: {detected_language}")
        else:
            status_label.config(text="No speech detected.")
    except Exception as e:
        # Display any errors that occur during processing
        messagebox.showerror("Error", str(e))
    finally:
        # Reset the status label to indicate readiness
        status_label.config(text="Ready")


# Function to start the speech processing in a separate thread
def start_speech_processing():
    threading.Thread(target=process_speech_to_language).start()


# -------------------- Build the LinguaLive GUI --------------------

# Create the main application window
app_window = tk.Tk()
app_window.title("LinguaLive: Real-Time Speech Recognition and Language Detection System")
app_window.geometry("600x400")
app_window.configure(bg="#f0f8ff")

# Add a title label
title_label = tk.Label(app_window, text="LinguaLive", font=("Arial", 16, "bold"), bg="#f0f8ff", fg="#000080")
title_label.pack(pady=10)

# Add a frame to hold the status and results
frame = tk.Frame(app_window, bg="#f0f8ff")
frame.pack(pady=10)

# Add a status label to show the current status of the application
status_label = tk.Label(frame, text="Ready", font=("Arial", 12), bg="#f0f8ff", fg="#006400")
status_label.pack(pady=5)

# Add a button to start capturing and detecting speech
start_button = tk.Button(frame, text="Start Speech Detection", font=("Arial", 12), bg="#000080", fg="#ffffff", command=start_speech_processing)
start_button.pack(pady=10)

# Add a label to display the recognized text
detected_text_label = tk.Label(frame, text="Recognized Text: ", font=("Arial", 12), wraplength=400, justify="left", bg="#f0f8ff", fg="#000000")
detected_text_label.pack(pady=10)

# Add a label to display the detected language
detected_language_label = tk.Label(frame, text="Detected Language: ", font=("Arial", 12), bg="#f0f8ff", fg="#000000")
detected_language_label.pack(pady=10)

# Run the GUI event loop
app_window.mainloop()