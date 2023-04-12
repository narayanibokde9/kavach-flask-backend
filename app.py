from flask import Flask, jsonify
from flask_mysqldb import MySQL
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PORT'] = 3310
app.config['MYSQL_PASSWORD'] = 'Sharayu@2000'
app.config['MYSQL_DB'] = 'spam_alert_system'

mysql = MySQL(app)


filename = "smsmodel.pkl"
with open(filename, 'rb') as f:
    model, tokenizer = pickle.load(f)

with open('emailmodel.pkl', 'rb') as f:
    clf = pickle.load(f)


def predictSMS(text_data):
    # Tokenize the input text data
    tokenized_data = tokenizer(text_data, padding=True, return_tensors="pt")

    # Pass the tokenized data through the model and get the predicted labels
    output_prediction = model(**tokenized_data)
    numpy_output = output_prediction.logits.detach().numpy()
    preds = np.argmax(numpy_output, axis=-1)

    # Return the predicted labels
    return preds


# This is for the email part
def predictEmail(emails):
    # Use the loaded pipeline to make predictions
    predictions = clf.predict(emails)
    return list(predictions)


text_data = "Congratulations! You have been selected as a winner. Text WIN to 55555 to claim your prize."
predicted_labels = predictSMS(text_data)
# print(predicted_labels)

# This is the usage of the email part
email = {
    "emailId": "example@example.com",
    "content": "Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES"
}
sms = {
    "phoneNo": "1234567890",
    "content": "Congratulations! You have been selected as a winner. Text WIN to 55555 to claim your prize."
}
predictions = predictEmail(email)
# print(predictions)


@app.route('/emails', methods=['GET', 'POST'])
def emails():
    data = request.get_json()
    content = data['content']
    result = predictEmail(content)
    return jsonify(result)


@app.route('/sms',  methods=['GET', 'POST'])
def sms():
    data = request.get_json()
    content = data['content']
    result = predictEmail(content)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

# emails --> SEND EMAIL TEXT TO ML MODEL
# sms --> SEND SMS TEXT TO ML MODEL
# call --> QUERY PHONE NO.  --> IF IT IS THERE IN DATABASE --> CHECK SPAM VOTES AND SEND RISK SCORE JSON
# IF NOT THERE IN DATABASE --> ADD PHONE NO. --> THEN SEND "Unknown Number"
