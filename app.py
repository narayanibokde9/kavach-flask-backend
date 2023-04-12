from flask import Flask, jsonify
from flask_mysqldb import MySQL
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'spam_alert_system'

mysql = MySQL(app)


filename = "smsmodel.pkl"
with open(filename, 'rb') as f:
    smsmodel, tokenizer = pickle.load(f)

with open('emailmodel.pkl', 'rb') as f:
    clf = pickle.load(f)


def predictSMS(text_data):
    # Tokenize the input text data
    tokenized_data = tokenizer(text_data, padding=True, return_tensors="pt")

    # Pass the tokenized data through the model and get the predicted labels
    output_prediction = smsmodel(**tokenized_data)
    numpy_output = output_prediction.logits.detach().numpy()
    preds = np.argmax(numpy_output, axis=-1)

    # Return the predicted labels
    return preds


# This is for the email part
def predictEmail(emails):
    # Use the loaded pipeline to make predictions
    predictions = clf.predict(emails)
    return list(predictions)


# This is the usage of the email part
email = {
    "emailId": "example@example.com",
    "content": "Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES"
}
sms = {
    "phoneNo": "1234567890",
    "content": "Congratulations! You have been selected as a winner. Text WIN to 55555 to claim your prize."
}


# 1. Get request -- get spam count from database


@app.route('/phoneSpamCount/<int:phone_no>', methods=['POST', 'GET'])
def phoneSpamCount(phone_no):
    cur = mysql.connection.cursor()
    query = "SELECT up_votes, down_votes FROM phone_calls WHERE phone_no = %s"
    cur.execute(query, (phone_no,))
    result = cur.fetchone()
    # Check if the phone number was found in the database
    if result is None:
        cur.execute("INSERT INTO phone_calls VALUES (%s, %s, %s)",
                    (phone_no, 0, 0))
        mysql.connection.commit()
        return jsonify({"error": "Phone number not found"})
    # for value in result:
    up_votes, down_votes = result
    spam_risk = 0
    # Create a JSON response with the up_votes and down_votes data
    response = {
        "phone_no": phone_no,
        "up_votes": up_votes,
        "down_votes": down_votes,
        "spam_risk": spam_risk
    }
    try:
        spam_risk = (down_votes/(down_votes+up_votes))
        return jsonify(response)
    except:
        spam_risk = 0
        return jsonify(response)


# 2. Post request -- post vote to database


@app.route('/phoneCallVote/<int:phone_no>/vote', methods=['POST', 'GET'])
def phoneCallVote(phone_no):
    upvote = request.args.get('upvote')
    # downvote = request.args.get('downvote')

    cur = mysql.connection.cursor()
    query = "SELECT up_votes, down_votes FROM phone_calls WHERE phone_no = %s"
    cur.execute(query, (phone_no,))
    result = cur.fetchone()

    # Check if the phone number was found in the database
    if result is None:
        return jsonify({"error": "Phone number not found"})

    up_votes, down_votes = result

    if upvote == '1':
        up_votes += 1
    elif upvote == '-1':
        down_votes += 1

    # Update the database with the new vote counts
    update_query = "UPDATE phone_calls SET up_votes=%s, down_votes=%s WHERE phone_no=%s"
    cur.execute(update_query, (up_votes, down_votes, phone_no))

    # Commit the changes to the database
    mysql.connection.commit()

    spam_risk = (down_votes/(down_votes+up_votes))
    # Create a JSON response with the up_votes and down_votes data
    response = {
        "phone_no": phone_no,
        "up_votes": up_votes,
        "down_votes": down_votes,
        "spam_risk": spam_risk
    }

    return jsonify(response)


@app.route('/email', methods=['POST'])
async def email():
    data = request.get_json()
    content = []
    content.insert(0, data['content'])
    prediction = predictEmail(content)
    print(prediction)
    email_id = data['emailId']

    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT email_id FROM email where email_id = %s", (email_id,))

    senderData = cur.fetchone()

    if senderData is None:
        cur.execute(
            "INSERT INTO email (email_id) VALUES (%s)", (email_id,))
    cur.execute(
        "SELECT email_id, no_of_reports,total_invokations FROM email where email_id = %s", (email_id,))
    senderData = cur.fetchone()
    email_id, no_of_reports, total_invokations = senderData
    if prediction[0] == 1:
        no_of_reports += 1
    total_invokations += 1
    update_query = "UPDATE email SET no_of_reports=%s, total_invokations=%s WHERE email_id=%s"
    cur.execute(update_query, (no_of_reports,
                               total_invokations, email_id,))
    mysql.connection.commit()
    result = {
        "isSpam": int(prediction[0]),
        "no_of_reports": no_of_reports,
    }
    cur.close()
    return jsonify(result)


@app.route('/sms',  methods=['GET', 'POST'])
def sms():
    data = request.get_json()
    content = data['content']
    result = predictEmail(content)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
