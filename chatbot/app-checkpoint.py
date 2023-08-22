{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e0e4d5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, render_template, request\n",
    "import openai\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "bfe4725a",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = Flask(__name__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f093a28d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set up your OpenAI API key\n",
    "openai.api_key =  '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'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d51ae3a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "data = None  # Initialize the variable to hold the dataset\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4dfa4c0d",
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.route('/')\n",
    "def index():\n",
    "    return render_template('index.html')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c4e704de",
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.route('/chatbot', methods=['POST'])\n",
    "def chatbot():\n",
    "    user_input = request.form['user-input']\n",
    "    response = chat_with_bot(user_input)\n",
    "    return response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "00c54df7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enter the path to your dataset file: C:/Users/subhanshu.pant/Downloads/customer_data.csv\n"
     ]
    }
   ],
   "source": [
    "# # Load the dataset\n",
    "# while True:\n",
    "#     file_path = input(\"Enter the path to your dataset file: \")\n",
    "#     try:\n",
    "#         data = pd.read_csv(file_path)\n",
    "#         break\n",
    "#     except FileNotFoundError:\n",
    "#         print(\"File not found. Please enter a valid file path.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4125faf2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define data quality rules and checks\n",
    "def check_completeness(data, attribute):\n",
    "    missing_values = data[attribute].isnull().sum()\n",
    "    if missing_values <= 0:\n",
    "        alert = f\"Data quality problem: Missing values found in attribute '{attribute}': {missing_values}\"\n",
    "        return alert\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8b44118c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def check_consistency(data, attribute):\n",
    "    unique_values = data[attribute].nunique()\n",
    "    if unique_values > 1:\n",
    "        alert = f\"Data quality problem: Inconsistent values found in attribute '{attribute}'\"\n",
    "        return alert\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "bd150895",
   "metadata": {},
   "outputs": [],
   "source": [
    "def check_formatting(data, attribute, regex_pattern):\n",
    "    # Check for formatting based on a regular expression pattern\n",
    "    data[attribute] = data[attribute].astype(str)  # Convert attribute to string type\n",
    "    incorrect_format_count = data[~data[attribute].str.match(regex_pattern, na=False)].shape[0]\n",
    "    if incorrect_format_count > 0:\n",
    "        alert = f\"Incorrect formatting found in '{attribute}': {incorrect_format_count}\"\n",
    "        return alert"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a867cf60",
   "metadata": {},
   "outputs": [],
   "source": [
    "def check_outliers(data, attribute, threshold):\n",
    "    # Check for outliers based on a threshold\n",
    "    data[attribute] = pd.to_numeric(data[attribute], errors='coerce')  # Convert attribute to numeric type\n",
    "    outliers = data[(data[attribute] < threshold[0]) | (data[attribute] > threshold[1])]\n",
    "    if not outliers.empty:\n",
    "        alert = f\"Outliers found in '{attribute}':\\n{outliers}\"\n",
    "        return alert\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a20b250c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to interact with the ChatGPT model\n",
    "def chat_with_bot(user_input):\n",
    "    if 'data quality' in user_input:\n",
    "        data_quality_alerts = evaluate_data_quality(data)\n",
    "        if data_quality_alerts:\n",
    "            return '\\n'.join(data_quality_alerts)\n",
    "        else:\n",
    "            return \"No data quality problems found.\"\n",
    "    else:\n",
    "        response = openai.Completion.create(\n",
    "            engine='davinci',\n",
    "            prompt=user_input,\n",
    "            max_tokens=100,\n",
    "            n=1,\n",
    "            stop=None,\n",
    "            temperature=0.7\n",
    "        )\n",
    "        return response.choices[0].text.strip()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "0ec7b02a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Main function for data quality evaluation\n",
    "def evaluate_data_quality(data):\n",
    "    alerts = []\n",
    "    attributes_to_check = data.columns\n",
    "\n",
    "    for attribute in attributes_to_check:\n",
    "        completeness_alert = check_completeness(data, attribute)\n",
    "        if completeness_alert:\n",
    "            alerts.append(completeness_alert)\n",
    "\n",
    "        consistency_alert = check_consistency(data, attribute)\n",
    "        if consistency_alert:\n",
    "            alerts.append(consistency_alert)\n",
    "\n",
    "        # Implement checks for formatting and outliers\n",
    "        if attribute == \"date\":\n",
    "            regex_pattern = r'^\\d{2}-\\d{2}-\\d{4}$'  # dd-mm-yyyy format\n",
    "        elif attribute == [\"amount\",\"price\"]:\n",
    "            regex_pattern = r'^\\d+(\\.\\d{1,2})?$'  # numeric format with up to 2 decimal places\n",
    "        elif attribute == [\"contact no\",\"phone no.\",\"mobile no.\"]:\n",
    "            regex_pattern = r'/^([+]\\d{2})?\\d{10}$/'  # contact number format\n",
    "        else:\n",
    "            regex_pattern = r''  # Add custom formatting pattern for other columns\n",
    "\n",
    "        formatting_alert = check_formatting(data, attribute, regex_pattern)\n",
    "        if formatting_alert:\n",
    "            alerts.append(formatting_alert)\n",
    "\n",
    "        threshold = (0, 100)  # Add your threshold values for outliers check\n",
    "        outliers_alert = check_outliers(data, attribute, threshold)\n",
    "        if outliers_alert:\n",
    "            alerts.append(outliers_alert)\n",
    "\n",
    "    return alerts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c09c1b7c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ... Rest of the code ...\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    # Load the dataset here\n",
    "    while True:\n",
    "        file_path = input(\"Enter the path to your dataset file: \")\n",
    "        try:\n",
    "            data = pd.read_csv(file_path)\n",
    "            break\n",
    "        except FileNotFoundError:\n",
    "            print(\"File not found. Please enter a valid file path.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fe04cb90",
   "metadata": {},
   "outputs": [],
   "source": [
    " app.run()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd40081d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "User: Describe the data quality in the form of regex pattern?\n",
      "Chatbot: Data quality problem: Missing values found in attribute 'invoice_no': 0\n",
      "Data quality problem: Inconsistent values found in attribute 'invoice_no'\n",
      "Data quality problem: Missing values found in attribute 'customer_id': 0\n",
      "Data quality problem: Inconsistent values found in attribute 'customer_id'\n",
      "Data quality problem: Missing values found in attribute 'gender': 0\n",
      "Data quality problem: Inconsistent values found in attribute 'gender'\n",
      "Data quality problem: Missing values found in attribute 'age': 0\n",
      "Data quality problem: Inconsistent values found in attribute 'age'\n",
      "Data quality problem: Missing values found in attribute 'category': 0\n",
      "Data quality problem: Inconsistent values found in attribute 'category'\n",
      "Data quality problem: Missing values found in attribute 'quantity': 0\n",
      "Data quality problem: Inconsistent values found in attribute 'quantity'\n",
      "Data quality problem: Missing values found in attribute 'price': 0\n",
      "Data quality problem: Inconsistent values found in attribute 'price'\n",
      "Outliers found in 'price':\n",
      "       invoice_no  customer_id  gender  age  category  quantity    price  \\\n",
      "0             NaN          NaN     NaN   28       NaN         5  1500.40   \n",
      "1             NaN          NaN     NaN   21       NaN         3  1800.51   \n",
      "2             NaN          NaN     NaN   20       NaN         1   300.08   \n",
      "3             NaN          NaN     NaN   66       NaN         5  3000.85   \n",
      "5             NaN          NaN     NaN   28       NaN         5  1500.40   \n",
      "...           ...          ...     ...  ...       ...       ...      ...   \n",
      "99447         NaN          NaN     NaN   37       NaN         3   107.52   \n",
      "99448         NaN          NaN     NaN   65       NaN         4  2400.68   \n",
      "99449         NaN          NaN     NaN   65       NaN         1   300.08   \n",
      "99451         NaN          NaN     NaN   50       NaN         5   179.20   \n",
      "99455         NaN          NaN     NaN   56       NaN         4  4200.00   \n",
      "\n",
      "      payment_method invoice_date   shopping_mall  \n",
      "0        Credit Card     5/8/2022          Kanyon  \n",
      "1         Debit Card   12/12/2021  Forum Istanbul  \n",
      "2               Cash    9/11/2021       Metrocity  \n",
      "3        Credit Card   16/05/2021    Metropol AVM  \n",
      "5        Credit Card   24/05/2022  Forum Istanbul  \n",
      "...              ...          ...             ...  \n",
      "99447           Cash   21/02/2021    Metropol AVM  \n",
      "99448    Credit Card   29/08/2021    Metropol AVM  \n",
      "99449           Cash     1/1/2023          Kanyon  \n",
      "99451           Cash    9/10/2021    Metropol AVM  \n",
      "99455           Cash   16/03/2021    Istinye Park  \n",
      "\n",
      "[64783 rows x 10 columns]\n",
      "Data quality problem: Missing values found in attribute 'payment_method': 0\n",
      "Data quality problem: Inconsistent values found in attribute 'payment_method'\n",
      "Data quality problem: Missing values found in attribute 'invoice_date': 0\n",
      "Data quality problem: Inconsistent values found in attribute 'invoice_date'\n",
      "Data quality problem: Missing values found in attribute 'shopping_mall': 0\n",
      "Data quality problem: Inconsistent values found in attribute 'shopping_mall'\n",
      "****************************************************************************************************\n"
     ]
    }
   ],
   "source": [
    "# # Interact with the chatbot\n",
    "# while True:\n",
    "#     user_input = input(\"User: \")\n",
    "#     response = chat_with_bot(user_input)\n",
    "#     print(\"Chatbot:\", response)\n",
    "#     print(\"*\"*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "689be41c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03cbd972",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "917f3b61",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
