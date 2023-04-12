from flask import Flask, jsonify
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



# @app.route('/get')

if __name__ == '__main__':
    app.run(debug=True)
