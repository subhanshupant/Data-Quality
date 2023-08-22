{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
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
   "id": "c889638a",
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
   "execution_count": 4,
   "id": "f07403dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "data = None  # Initialize the variable to hold the dataset\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "5cc9e699",
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
   "execution_count": 6,
   "id": "49eb341a",
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
   "execution_count": 7,
   "id": "00c54df7",
   "metadata": {},
   "outputs": [],
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
   "execution_count": 8,
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
   "execution_count": 9,
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
   "execution_count": 10,
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
   "execution_count": 11,
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
   "execution_count": 12,
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
   "execution_count": 13,
   "id": "3cf45965",
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
   "execution_count": 14,
   "id": "4be44621",
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
   "execution_count": 19,
   "id": "49368313",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "\u001b[31m   WARNING: This is a development server. Do not use it in a production deployment.\u001b[0m\n",
      "\u001b[2m   Use a production WSGI server instead.\u001b[0m\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n",
      "[2023-05-27 17:42:23,897] ERROR in app: Exception on / [GET]\n",
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"C:\\Users\\subhanshu.pant\\AppData\\Local\\Temp\\ipykernel_29928\\1849266281.py\", line 3, in index\n",
      "    return render_template('index.html')\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\templating.py\", line 138, in render_template\n",
      "    ctx.app.jinja_env.get_or_select_template(template_name_or_list),\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\jinja2\\environment.py\", line 930, in get_or_select_template\n",
      "    return self.get_template(template_name_or_list, parent, globals)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\jinja2\\environment.py\", line 883, in get_template\n",
      "    return self._load_template(name, self.make_globals(globals))\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\jinja2\\environment.py\", line 857, in _load_template\n",
      "    template = self.loader.load(self, name, globals)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\jinja2\\loaders.py\", line 115, in load\n",
      "    source, filename, uptodate = self.get_source(environment, name)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\templating.py\", line 60, in get_source\n",
      "    return self._get_source_fast(environment, template)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\templating.py\", line 89, in _get_source_fast\n",
      "    raise TemplateNotFound(template)\n",
      "jinja2.exceptions.TemplateNotFound: index.html\n",
      "127.0.0.1 - - [27/May/2023 17:42:23] \"GET / HTTP/1.1\" 500 -\n",
      "[2023-05-27 17:42:26,594] ERROR in app: Exception on / [GET]\n",
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"C:\\Users\\subhanshu.pant\\AppData\\Local\\Temp\\ipykernel_29928\\1849266281.py\", line 3, in index\n",
      "    return render_template('index.html')\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\templating.py\", line 138, in render_template\n",
      "    ctx.app.jinja_env.get_or_select_template(template_name_or_list),\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\jinja2\\environment.py\", line 930, in get_or_select_template\n",
      "    return self.get_template(template_name_or_list, parent, globals)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\jinja2\\environment.py\", line 883, in get_template\n",
      "    return self._load_template(name, self.make_globals(globals))\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\jinja2\\environment.py\", line 857, in _load_template\n",
      "    template = self.loader.load(self, name, globals)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\jinja2\\loaders.py\", line 115, in load\n",
      "    source, filename, uptodate = self.get_source(environment, name)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\templating.py\", line 60, in get_source\n",
      "    return self._get_source_fast(environment, template)\n",
      "  File \"C:\\Users\\subhanshu.pant\\Anaconda3\\lib\\site-packages\\flask\\templating.py\", line 89, in _get_source_fast\n",
      "    raise TemplateNotFound(template)\n",
      "jinja2.exceptions.TemplateNotFound: index.html\n",
      "127.0.0.1 - - [27/May/2023 17:42:26] \"GET / HTTP/1.1\" 500 -\n"
     ]
    }
   ],
   "source": [
    " app.run()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "dd40081d",
   "metadata": {},
   "outputs": [],
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
   "execution_count": None,
   "id": "689be41c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "5bf78de5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "User: Describe the quality of the dataset\n",
      "Chatbot: The dataset is of high quality as it includes relevant information such as invoice number, customer ID, gender, age, category, quantity, price, payment method, invoice date, and shopping mall. This data can be used to draw insights on customer behaviour and preferences.\n",
      "****************************************************************************************************\n"
     ]
    }
   ],
   "source": [
    "# Function to interact with the ChatGPT model\n",
    "import pandas as pd\n",
    "from tabulate import tabulate\n",
    "import openai\n",
    "\n",
    "openai.api_key =  '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'\n",
    "\n",
    "dataset=pd.read_csv(r\"C:\\Users\\subhanshu.pant\\Downloads\\customer_data.csv\")\n",
    "dataset1=dataset.head(20)\n",
    "table_text = tabulate(dataset1, headers= 'keys', tablefmt=\"psql\")\n",
    "                    \n",
    "def chat_with_bot(user_input):\n",
    "    prompt=\"you are a inelligent assistant. you can use the table to draw insight.\" + table_text + user_input\n",
    "    response = openai.Completion.create(\n",
    "        engine='text-davinci-003',\n",
    "        prompt=prompt,\n",
    "        max_tokens=100,\n",
    "        n=1,\n",
    "        stop=None,\n",
    "        temperature=0.7\n",
    "    )\n",
    "    return response.choices[0].text.strip()\n",
    "    \n",
    "    \n",
    "def input_to_bot():\n",
    "    while True:\n",
    "        user_input = input(\"User: \")\n",
    "        if user_input == 'exit':\n",
    "            break\n",
    "        else:\n",
    "            response = chat_with_bot(user_input)\n",
    "            print(\"Chatbot:\", response)\n",
    "            print(\"*\"*100)\n",
    "            \n",
    "input_to_bot()\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "8c95ed4b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: gradio in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (3.32.0)\n",
      "Requirement already satisfied: markdown-it-py[linkify]>=2.0.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (2.2.0)\n",
      "Requirement already satisfied: semantic-version in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (2.10.0)\n",
      "Requirement already satisfied: aiofiles in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (23.1.0)\n",
      "Requirement already satisfied: numpy in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (1.21.5)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (2.11.3)\n",
      "Requirement already satisfied: pydantic in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (1.10.8)\n",
      "Requirement already satisfied: pyyaml in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (6.0)\n",
      "Requirement already satisfied: websockets>=10.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (11.0.3)\n",
      "Requirement already satisfied: matplotlib in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (3.5.1)\n",
      "Requirement already satisfied: gradio-client>=0.2.4 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (0.2.5)\n",
      "Requirement already satisfied: uvicorn>=0.14.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (0.22.0)\n",
      "Requirement already satisfied: ffmpy in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (0.3.0)\n",
      "Requirement already satisfied: altair>=4.2.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (5.0.1)\n",
      "Requirement already satisfied: aiohttp in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (3.8.1)\n",
      "Requirement already satisfied: pandas in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (1.4.2)\n",
      "Requirement already satisfied: orjson in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (3.8.14)\n",
      "Requirement already satisfied: markupsafe in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (2.0.1)\n",
      "Requirement already satisfied: python-multipart in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (0.0.6)\n",
      "Requirement already satisfied: pydub in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (0.25.1)\n",
      "Requirement already satisfied: pygments>=2.12.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (2.15.1)\n",
      "Requirement already satisfied: fastapi in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (0.95.2)\n",
      "Requirement already satisfied: httpx in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (0.24.1)\n",
      "Requirement already satisfied: typing-extensions in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (4.6.2)\n",
      "Requirement already satisfied: huggingface-hub>=0.13.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (0.14.1)\n",
      "Requirement already satisfied: requests in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (2.27.1)\n",
      "Requirement already satisfied: mdit-py-plugins<=0.3.3 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (0.3.3)\n",
      "Requirement already satisfied: pillow in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio) (9.0.1)\n",
      "Requirement already satisfied: toolz in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from altair>=4.2.0->gradio) (0.11.2)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from altair>=4.2.0->gradio) (4.4.0)\n",
      "Requirement already satisfied: packaging in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio-client>=0.2.4->gradio) (21.3)\n",
      "Requirement already satisfied: fsspec in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from gradio-client>=0.2.4->gradio) (2022.2.0)\n",
      "Requirement already satisfied: filelock in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.13.0->gradio) (3.6.0)\n",
      "Requirement already satisfied: tqdm>=4.42.1 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.13.0->gradio) (4.64.0)\n",
      "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair>=4.2.0->gradio) (0.18.0)\n",
      "Requirement already satisfied: attrs>=17.4.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair>=4.2.0->gradio) (21.4.0)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from markdown-it-py[linkify]>=2.0.0->gradio) (0.1.2)\n",
      "Requirement already satisfied: linkify-it-py<3,>=1 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from markdown-it-py[linkify]>=2.0.0->gradio) (2.0.2)\n",
      "Requirement already satisfied: uc-micro-py in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from linkify-it-py<3,>=1->markdown-it-py[linkify]>=2.0.0->gradio) (1.0.2)\n",
      "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from packaging->gradio-client>=0.2.4->gradio) (3.0.4)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from pandas->gradio) (2021.3)\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from pandas->gradio) (2.8.2)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.1->pandas->gradio) (1.16.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from tqdm>=4.42.1->huggingface-hub>=0.13.0->gradio) (0.4.4)\n",
      "Requirement already satisfied: click>=7.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from uvicorn>=0.14.0->gradio) (8.0.4)\n",
      "Requirement already satisfied: h11>=0.8 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from uvicorn>=0.14.0->gradio) (0.14.0)\n",
      "Requirement already satisfied: frozenlist>=1.1.1 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from aiohttp->gradio) (1.2.0)\n",
      "Requirement already satisfied: charset-normalizer<3.0,>=2.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from aiohttp->gradio) (2.0.4)\n",
      "Requirement already satisfied: aiosignal>=1.1.2 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from aiohttp->gradio) (1.2.0)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from aiohttp->gradio) (1.6.3)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from aiohttp->gradio) (5.1.0)\n",
      "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from aiohttp->gradio) (4.0.1)\n",
      "Requirement already satisfied: idna>=2.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from yarl<2.0,>=1.0->aiohttp->gradio) (3.3)\n",
      "Requirement already satisfied: starlette<0.28.0,>=0.27.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from fastapi->gradio) (0.27.0)\n",
      "Requirement already satisfied: anyio<5,>=3.4.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from starlette<0.28.0,>=0.27.0->fastapi->gradio) (3.5.0)\n",
      "Requirement already satisfied: sniffio>=1.1 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from anyio<5,>=3.4.0->starlette<0.28.0,>=0.27.0->fastapi->gradio) (1.2.0)\n",
      "Requirement already satisfied: certifi in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from httpx->gradio) (2021.10.8)\n",
      "Requirement already satisfied: httpcore<0.18.0,>=0.15.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from httpx->gradio) (0.17.2)\n",
      "Requirement already satisfied: cycler>=0.10 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from matplotlib->gradio) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from matplotlib->gradio) (4.25.0)\n",
      "Requirement already satisfied: kiwisolver>=1.0.1 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from matplotlib->gradio) (1.3.2)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\users\\subhanshu.pant\\anaconda3\\lib\\site-packages (from requests->gradio) (1.26.9)\n"
     ]
    }
   ],
   "source": [
    "# !pip install gradio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "a1254cbd",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "dataclass_transform() got an unexpected keyword argument 'field_specifiers'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[1;32mIn [40]\u001b[0m, in \u001b[0;36m<cell line: 3>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mpandas\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mpd\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtabulate\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m tabulate\n\u001b[1;32m----> 3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mgr\u001b[39;00m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mopenai\u001b[39;00m\n\u001b[0;32m      6\u001b[0m openai\u001b[38;5;241m.\u001b[39mapi_key \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124msk-zz0wGMzrUKUOVvbGUh00T3BlbkFJTATYYMbGbAAYfR0EWlJi\u001b[39m\u001b[38;5;124m'\u001b[39m\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\gradio\\__init__.py:3\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mpkgutil\u001b[39;00m\n\u001b[1;32m----> 3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcomponents\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mcomponents\u001b[39;00m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01minputs\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01minputs\u001b[39;00m\n\u001b[0;32m      5\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mgradio\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01moutputs\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01moutputs\u001b[39;00m\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\gradio\\components.py:32\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     30\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mPIL\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mImageOps\u001b[39;00m\n\u001b[0;32m     31\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mrequests\u001b[39;00m\n\u001b[1;32m---> 32\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m UploadFile\n\u001b[0;32m     33\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mffmpy\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m FFmpeg\n\u001b[0;32m     34\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mgradio_client\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m media_data\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\fastapi\\__init__.py:7\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      3\u001b[0m __version__ \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m0.95.2\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m      5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mstarlette\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m status \u001b[38;5;28;01mas\u001b[39;00m status\n\u001b[1;32m----> 7\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mapplications\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m FastAPI \u001b[38;5;28;01mas\u001b[39;00m FastAPI\n\u001b[0;32m      8\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mbackground\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m BackgroundTasks \u001b[38;5;28;01mas\u001b[39;00m BackgroundTasks\n\u001b[0;32m      9\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdatastructures\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m UploadFile \u001b[38;5;28;01mas\u001b[39;00m UploadFile\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\fastapi\\applications.py:16\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01menum\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Enum\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtyping\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m (\n\u001b[0;32m      3\u001b[0m     Any,\n\u001b[0;32m      4\u001b[0m     Awaitable,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     13\u001b[0m     Union,\n\u001b[0;32m     14\u001b[0m )\n\u001b[1;32m---> 16\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m routing\n\u001b[0;32m     17\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdatastructures\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Default, DefaultPlaceholder\n\u001b[0;32m     18\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mencoders\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m DictIntStrAny, SetIntStr\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\fastapi\\routing.py:24\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     22\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m params\n\u001b[0;32m     23\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdatastructures\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Default, DefaultPlaceholder\n\u001b[1;32m---> 24\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdependencies\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Dependant\n\u001b[0;32m     25\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mdependencies\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mutils\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m (\n\u001b[0;32m     26\u001b[0m     get_body_field,\n\u001b[0;32m     27\u001b[0m     get_dependant,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m     30\u001b[0m     solve_dependencies,\n\u001b[0;32m     31\u001b[0m )\n\u001b[0;32m     32\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mencoders\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m DictIntStrAny, SetIntStr, jsonable_encoder\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\fastapi\\dependencies\\models.py:3\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtyping\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Any, Callable, List, Optional, Sequence\n\u001b[1;32m----> 3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01msecurity\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mbase\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m SecurityBase\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mpydantic\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mfields\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m ModelField\n\u001b[0;32m      7\u001b[0m \u001b[38;5;28;01mclass\u001b[39;00m \u001b[38;5;21;01mSecurityRequirement\u001b[39;00m:\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\fastapi\\security\\__init__.py:1\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mapi_key\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m APIKeyCookie \u001b[38;5;28;01mas\u001b[39;00m APIKeyCookie\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mapi_key\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m APIKeyHeader \u001b[38;5;28;01mas\u001b[39;00m APIKeyHeader\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mapi_key\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m APIKeyQuery \u001b[38;5;28;01mas\u001b[39;00m APIKeyQuery\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\fastapi\\security\\api_key.py:3\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtyping\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Optional\n\u001b[1;32m----> 3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mopenapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m APIKey, APIKeyIn\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01msecurity\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mbase\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m SecurityBase\n\u001b[0;32m      5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mstarlette\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mexceptions\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m HTTPException\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\fastapi\\openapi\\models.py:5\u001b[0m, in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtyping\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Any, Callable, Dict, Iterable, List, Optional, Union\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mfastapi\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mlogger\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m logger\n\u001b[1;32m----> 5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mpydantic\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m AnyUrl, BaseModel, Field\n\u001b[0;32m      7\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m      8\u001b[0m     \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01memail_validator\u001b[39;00m  \u001b[38;5;66;03m# type: ignore\u001b[39;00m\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pydantic\\__init__.py:2\u001b[0m, in \u001b[0;36minit pydantic.__init__\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pydantic\\dataclasses.py:48\u001b[0m, in \u001b[0;36minit pydantic.dataclasses\u001b[1;34m()\u001b[0m\n",
      "File \u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pydantic\\main.py:120\u001b[0m, in \u001b[0;36minit pydantic.main\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: dataclass_transform() got an unexpected keyword argument 'field_specifiers'"
     ]
    }
   ],
   "source": [
    "# import pandas as pd\n",
    "# from tabulate import tabulate\n",
    "# import gradio as gr\n",
    "# import openai\n",
    "\n",
    "# openai.api_key = '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'\n",
    "\n",
    "# dataset = pd.read_csv(r\"C:\\Users\\subhanshu.pant\\Downloads\\customer_data.csv\")\n",
    "# dataset1 = dataset.head(20)\n",
    "# table_text = tabulate(dataset1, headers='keys', tablefmt=\"psql\")\n",
    "\n",
    "\n",
    "# def chat_with_bot(user_input):\n",
    "#     prompt = \"you are an intelligent assistant. you can use the table to draw insight.\\n\" + table_text + \"\\n\" + user_input\n",
    "#     response = openai.Completion.create(\n",
    "#         engine='text-davinci-003',\n",
    "#         prompt=prompt,\n",
    "#         max_tokens=100,\n",
    "#         n=1,\n",
    "#         stop=None,\n",
    "#         temperature=0.7\n",
    "#     )\n",
    "#     return response.choices[0].text.strip()\n",
    "\n",
    "\n",
    "# iface = gr.Interface(fn=chat_with_bot, inputs=\"text\", outputs=\"text\", title=\"Chat with Intelligent Assistant\")\n",
    "# iface.launch()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "f9f9dc36",
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
