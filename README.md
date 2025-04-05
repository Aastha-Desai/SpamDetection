Spam Detection Using Machine Learning
This project uses machine learning to detect spam emails. The model is trained on a dataset of labeled emails (spam or not spam), and a Streamlit application is provided to interact with the model in a web interface.

Files in This Repository
train_model.py - Python script to train the machine learning model using the email spam dataset.

app.py - Streamlit app that allows users to input email text and get spam predictions.

spam_model.pkl - Trained machine learning model saved for later use in the Streamlit app.

spam.csv - The CSV file used for training the model (replace with your own dataset if needed).

requirements.txt - A file that lists the Python dependencies required to run the project.

Setup Instructions
1. Clone the Repository
- git clone https://github.com/Aastha-Desai/SpamDetection.git
- cd SpamDetection
2. Install Dependencies
Before you start using the project, make sure to install the required Python packages listed in requirements.txt. You can do this using pip:
- pip install -r requirements.txt
3. Train the Model
To train the model, you need to run the train_model.py script. This will load the dataset, train a machine learning model, and save the trained model as a .pkl file.
- python train_model.py
4. Run the Streamlit Application
Once the model is trained, you can run the Streamlit app by using the following command:
- streamlit run app.py
This will start a local web server where you can interact with the model by entering text from an email and predicting whether it is spam or not.

5. Using the App
Open the Streamlit app in your browser (usually at http://localhost:8501).

Enter text from an email (spam or not spam) into the input field and click on the "Predict" button to get the result.

Dataset
This model is trained using a Kaggle email spam dataset, but you can replace it with your own dataset. The dataset must be in CSV format and contain two columns: label (spam or not spam) and text (email content).

Example Dataset:
label	text
Not Spam	Subject: Meeting at 2pm
spam	Congratulations! You've won!

This is the spam detector:
https://spamdetection-email.streamlit.app/ 
