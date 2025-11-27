# **LinguaLive: Real-Time Language Detection System**

Hello! Welcome to LinguaLive, a project developed for the CSCI 2304 Intelligent Systems course at the International Islamic University Malaysia (IIUM).

LinguaLive is designed to break down communication barriers by providing real-time language detection from spoken input. It uses a machine learning pipeline to quickly and accurately determine the language being spoken.

## 🚀 Key Features

1. Real-Time Speech Recognition: Converts spoken input into text using the SpeechRecognition library.
2. Machine Learning Classification: Uses a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer combined with a Multinomial Naive Bayes classifier to predict the language of the transcribed text.
3. User-Friendly GUI: Implemented using the Tkinter library for simple, interactive operation.
4. 17 Supported Languages: The model was trained on a dataset covering languages including English, French, Spanish, German, Arabic, Hindi, and more

## 💻 How It Works

The system follows a simple process:
1. Speech Input: The user clicks the "Start Speech Detection" button and speaks into their microphone.
2. Conversion: The SpeechRecognition library converts the audio input into raw text.
3.Language Prediction: The text is passed to the trained machine learning model, which uses weighted word frequencies (TF-IDF) to quickly classify the text and predict the correct language (Multinomial Naive Bayes).
4.Output: The recognized text and the detected language are displayed instantly on the GUI.

## 🛠️ Setup and Installation

To run the application locally, you need Python and the required dependencies.
1. Install Libraries:
    'pip install pandas seaborn matplotlib scikit-learn SpeechRecognition PyAudio'
2. Run Application: Ensure the LINGUALIVE.py script and the dataset (Language-Detection.csv) are in the same folder.
3. Usage: The GUI will appear. Click "Start Speech Detection" to input audio.

## 💡 Future Works

Future efforts will focus on improving robustness in noisy environments, expanding language coverage, and handling mixed-language input (code-switching).

## 🧑‍💻 Team and their contribution

1. Nur Iman Amani Binti Ahmad Akhir - Methodology, ML model implementation, and analysis (https://github.com/ImanAkhir)
2. Fitrah Nur Humaira Binti Muhamad Radaudin - Coding, dataset selection, and deployment.
3. Zullaikha Binti Zulzahrin - Literature review, discussion, and future work summary.



## ***📝 Author***

***Fitrah Nur Humaira***

***This project is for educational purposes.***
