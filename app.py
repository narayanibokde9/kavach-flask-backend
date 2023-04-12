from flask import Flask, jsonify, request
from flask_mysqldb import MySQL

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PORT'] = 3310
app.config['MYSQL_PASSWORD'] = 'Sharayu@2000'
app.config['MYSQL_DB'] = 'spam_alert_system'
 
mysql = MySQL(app)

@app.route('/emails')
def emails():
    cur = mysql.connection.cursor()
    # cur.execute("INSERT INTO email VALUES (1, 'person1@gmail.com', 12, 0, 12), (2, 'person2@gmail.com', 112, 1, 28)")
    cur.execute("SELECT * FROM email")
    data = cur.fetchall()
    mysql.connection.commit()
    cur.close()
    return jsonify(data)

#1. Get request -- get spam count from database
@app.route('/phoneSpamCount/<int:phone_no>', methods = ['POST', 'GET'])
def phoneSpamCount(phone_no):
    cur = mysql.connection.cursor()
    print(phone_no)
    query = "SELECT up_votes, down_votes FROM phone_calls WHERE phone_no = %s"
    cur.execute(query, (phone_no,))
    result = cur.fetchone()
    # for value in result:
    print(result)
    up_votes, down_votes = result

    # Check if the phone number was found in the database
    if result is None:
        return jsonify({"error": "Phone number not found"})

    spam_risk = (down_votes/(down_votes+up_votes))
    # Create a JSON response with the up_votes and down_votes data
    response = {
        "phone_no": phone_no,
        "up_votes": up_votes,
        "down_votes": down_votes,
        "spam_risk" : spam_risk
    }

    return jsonify(response)

#2. Post request -- post vote to database
@app.route('/phoneCallVote/<int:phone_no>/vote', methods = ['POST', 'GET'])
def phoneCallVote(phone_no):
    upvote = request.args.get('upvote')
    downvote = request.args.get('downvote')

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
    elif downvote == '1':
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
        "spam_risk" : spam_risk
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)


