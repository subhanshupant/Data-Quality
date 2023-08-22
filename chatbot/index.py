from flask import Flask, render_template, request
import openai
import pandas as pd
from tabulate import tabulate

app = Flask(__name__)

# Set up your OpenAI API key
openai.api_key = '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'

# Load the dataset
data = None  # Initialize the variable to hold the dataset

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/templates/chatbot')
@app.route('/templates/about')
def chatbothtml():
    pageName = request.path[10:]
    return render_template(f'{pageName}.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.form['user-input']
    print(user_input)
    response = chat_with_bot(user_input)
    return response

@app.route('/upload', methods=['POST'])
def chatbotUpload():
    if 'file' in request.files:
        file = request.files['file']
        file.save(file.filename)
        try:
            global data
            data = pd.read_csv(file.filename)
            return evaluate_data_quality(data)
        except FileNotFoundError:
            return 'file not found'
    return 'File Uploaded'

# Define data quality rules and checks
def check_completeness(data, attribute):
    missing_values = data[attribute].isnull().sum()
    if missing_values >= 0:
        alert = f"Data quality problem: Missing values found in attribute '{attribute}': {missing_values}"
        return alert
    
def check_consistency(data, attribute):
    unique_values = data[attribute].nunique()
    if unique_values > 1:
        alert = f"Data quality problem: Inconsistent values found in attribute '{attribute}'"
        return alert
    
def check_formatting(data, attribute, regex_pattern):
    # Check for formatting based on a regular expression pattern
    data[attribute] = data[attribute].astype(str)  # Convert attribute to string type
    incorrect_format_count = data[~data[attribute].str.match(regex_pattern, na=False)].shape[0]
    if incorrect_format_count > 0:
        alert = f"Data quality problem: Incorrect formatting found in '{attribute}': {incorrect_format_count}"
        return alert

# def check_outliers(data, attribute, threshold):
#     # Check for outliers based on a threshold
#     data[attribute] = pd.to_numeric(data[attribute], errors='coerce')  # Convert attribute to numeric type
#     outliers = data[(data[attribute] < threshold[0]) | (data[attribute] > threshold[1])]
#     if not outliers.empty:
#         alert = f"Outliers found in '{attribute}':\n{outliers}"
#         return alert


def chat_with_bot(user_input):
    global data
    print(data)
    dataset = data.head(20)
    tableText = tabulate(dataset, headers='keys', tablefmt='psql')
    
    # Call evaluate_data_quality and assign the returned value to 'alerts'
    alerts = evaluate_data_quality(data)
    
    # Create the prompt with the dataset table and alerts
    prompt = f"you are an intelligent assistant. You can use the table to draw insight. '{tableText}'\n\n"
    if alerts:
        prompt += "Data quality problems have been detected:\n"
        prompt += "\n".join(alerts)
        prompt += "\n\n"
    prompt += f"User input: {user_input}"
    
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()
    
# Main function for data quality evaluation
def evaluate_data_quality(data):
    alerts = []
    attributes_to_check = data.columns

    for attribute in attributes_to_check:
        completeness_alert = check_completeness(data, attribute)
        if completeness_alert:
            alerts.append(completeness_alert)

        consistency_alert = check_consistency(data, attribute)
        if consistency_alert:
            alerts.append(consistency_alert)

        # Implement checks for formatting and outliers
        if attribute == "date":
            regex_pattern = r'^\d{2}-\d{2}-\d{4}$'  # dd-mm-yyyy format
        elif attribute == ["amount","price"]:
            regex_pattern = r'^\d+(\.\d{1,2})?$'  # numeric format with up to 2 decimal places
        elif attribute == ["contact no","phone no.","mobile no."]:
            regex_pattern = r'/^([+]\d{2})?\d{10}$/'  # contact number format
        else:
            regex_pattern = r''  # Add custom formatting pattern for other columns

        formatting_alert = check_formatting(data, attribute, regex_pattern)
        if formatting_alert:
            alerts.append(formatting_alert)

        # threshold = (0, 100)  # Add your threshold values for outliers check
        # outliers_alert = check_outliers(data, attribute, threshold)
        # if outliers_alert:
        #     alerts.append(outliers_alert)

    return alerts

# def input_to_bot():
#     while True:
#         if user_input == 'exit':
#             break
#         else:
#             response = chat_with_bot(user_input)
#             print("Chatbot:", response)
#             print("*"*100)
            
# input_to_bot()

# ... Rest of the code ...

if __name__ == '__main__':
    # Load the dataset here
    # while True:
    #     file_path = input("Enter the path to your dataset file: ")
    #     try:
    #         data = pd.read_csv(file_path)
    #         break
    #     except FileNotFoundError:
    #         print("File not found. Please enter a valid file path.")

    app.run()