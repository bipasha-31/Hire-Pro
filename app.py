# Import all the necessary libraries
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import re
import time
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from flask import Flask , render_template , request , url_for , jsonify , Response, session
from werkzeug.utils import redirect, secure_filename
from flask_mail import Mail , Message
from flask_mysqldb import MySQL
from pyresparser import ResumeParser
from fer import Video
from fer import FER
from video_analysis import extract_text , analyze_tone
from decouple import config
from datetime import datetime
import uuid
import os



# Access the environment variables stored in .env file
MYSQL_USER = config('mysql_user')
MYSQL_PASSWORD = config('mysql_password')

# To send mail (By interviewee)
MAIL_USERNAME = config('mail_username')
MAIL_PWD = config('mail_pwd')

# For logging into the interview portal
COMPANY_MAIL = config('company_mail')
COMPANY_PSWD = config('company_pswd')

# Create a Flask app
app = Flask(__name__)
app.secret_key='smart_hire_2025'

# App configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB'] = 'smart_hire' 
user_db = MySQL(app)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = MAIL_USERNAME
app.config['MAIL_PASSWORD'] = MAIL_PWD
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_ASCII_ATTACHMENTS'] = True
mail = Mail(app)

@app.before_request
def assign_session():
    if'session_id'not in session:
            session['session_id']=str(uuid.uuid4())



# Initial sliding page
@app.route('/')
def home():
    return render_template('index.html')



# Interviewee signup 

@app.route('/signup', methods=['POST', 'GET'])
def interviewee():
    if request.method == 'POST' and 'username' in request.form and 'usermail' in request.form and 'userpassword' in request.form:
        username = request.form['username']
        usermail = request.form['usermail']
        userpassword = request.form['userpassword']
        session_id=session['session_id']

        cursor = user_db.connection.cursor()

        # Check if account already exists with same username and email
        cursor.execute("SELECT * FROM candidates WHERE candidatename = %s AND email = %s", (username, usermail))
        account = cursor.fetchone()

        if account:
            err = "Account Already Exists"
            return render_template('index.html', err=err)

        # Validate email format
        elif not re.fullmatch(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', usermail):
            err = "Invalid Email Address !!"
            return render_template('index.html', err=err)

        # Validate username characters (only letters, numbers, spaces)
        elif not re.fullmatch(r'[A-Za-z0-9\s]+', username):
            err = "Username must contain only characters and numbers !!"
            return render_template('index.html', err=err)

        # Check if any field is empty
        elif not username or not userpassword or not usermail:
            err = "Please fill out all the fields"
            return render_template('index.html', err=err)

        else:
            # Insert new user record into database
            cursor.execute("INSERT INTO candidates(candidatename,email,password,session_id) VALUES (%s,%s,%s,%s)", (username, usermail, userpassword, session_id))
            user_db.connection.commit()
            reg = "You have successfully registered !!"
            return render_template('FirstPage.html', reg=reg)

    else:
        # For GET request, just render signup page
        return render_template('index.html')




# Interviewer signin 
@app.route('/signin', methods=['POST', 'GET'])
def interviewer():
    if request.method == 'POST':
        company_mail = request.form['company_mail']
        password = request.form['password']

        if company_mail == COMPANY_MAIL and password == COMPANY_PSWD:
            cursor = user_db.connection.cursor()
            cursor.execute("SELECT * FROM candidates")
            candidates = cursor.fetchall()
            return render_template('candidateSelect.html', candidates=candidates)
        else:
            return render_template("index.html", err="Incorrect Credentials")
    else:
        return render_template("index.html")



# personality trait prediction using Logistic Regression and parsing resume
# personality trait prediction using Logistic Regression and parsing resume
@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    # get form data
    if request.method == 'POST':
        session_id=session['session_id']
        # Fetch form fields and process them
        fname = request.form['firstname'].capitalize()
        lname = request.form['lastname'].capitalize()
        age = int(request.form['age'])  # Convert age to integer
        gender = request.form['gender']
        email = request.form['email']
        file = request.files['resume']
        
        # Save the resume to a path
        resume_path = f'./static/{session_id}_resume.pdf'
        file.save(resume_path)
        # Fetch numeric values for personality traits and convert them to integers
        try:
            val1 = int(request.form['openness'])  # Convert to int
            val2 = int(request.form['neuroticism'])
            val3 = int(request.form['conscientiousness'])
            val4 = int(request.form['agreeableness'])
            val5 = int(request.form['extraversion'])
        except ValueError:
            # In case any of the personality trait values are missing or invalid
            return "Error: Invalid input for personality traits. Please ensure all values are numeric.", 400

        # Prepare the data for prediction
        df = pd.read_csv(r'static\trainDataset.csv')
        
        # Encoding categorical gender values for the model
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        
        # Extract the feature and target columns for training the model
        x_train = df.iloc[:, :-1].to_numpy()  # Features
        y_train = df.iloc[:, -1].to_numpy(dtype=str)  # Target
        
        # Train the Logistic Regression model
        lreg = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        lreg.fit(x_train, y_train)

        # Map gender value to 1 (male) and 0 (female)
        if gender == 'male':
            gender = 1
        elif gender == 'female':
            gender = 0
        
        # Prepare input for prediction
        input_data = [gender, age, val1, val2, val3, val4, val5]
        
        # Predict the personality trait
        pred = str(lreg.predict([input_data])[0]).capitalize()

        # Extract data from the resume using ResumeParser
        data = ResumeParser(resume_path).get_extracted_data()

        # Create a dictionary with all the extracted data and prediction
        result = {
            'Name': fname + ' ' + lname,
            'Age': age,
            'Email': email,
            'Mobile Number': data.get('mobile_number', None),
            'Skills': str(data['skills']).replace("[", "").replace("]", "").replace("'", ""),
            'Degree': data.get('degree', None)[0] if data.get('degree', None) else None,
            'Designation': data.get('designation', None)[0] if data.get('designation', None) else None,
            'Total Experience': data.get('total_experience'),
            'Predicted Personality': pred
        }

        # Save the result as a JSON file
        with open(f'./static/{session_id}_result.json', 'w') as file:
            json.dump(result, file)

    return render_template('questionPage.html')



# Record candidate's interview for face emotion and tone analysis
@app.route('/analysis', methods=['POST'])
def video_analysis():
    session_id=session['session_id']


    # Save uploaded videos
    quest1 = request.files['question1']
    quest2 = request.files['question2']
    quest3 = request.files['question3']

    filenames=[f"{session_id}_question{i+1}.webm" for i in range(3)]
    paths=[f"./static/{fname}" for fname in filenames]
    for f,p in zip([quest1,quest2,quest3],paths):
        f.save(p)

    questions = [
        'Question 1: Tell something about yourself',
        'Question 2: Why should we hire you?',
        'Question 3: Where do you see yourself five years from now?'
    ]
    responses = {q: [] for q in questions}
    tone_results = []

    for i, fname in enumerate(filenames):
        text, _ = extract_text(fname)

        if not text.strip():
            print(f"[INFO] Skipping tone analysis for empty response to: {questions[i]}")
            tone_results.append([(tone, 0.0) for tone in ['anger', 'disgust', 'fear', 'joy', 'sadness']])
            continue

        responses[questions[i]].append(text)

        # Analyze tone
        try:
            res = analyze_tone(text)
            print(f"[DEBUG] Watson NLU emotion result for {questions[i]}: {res}")
        except Exception as e:
            print(f"[ERROR] NLU analysis failed for {questions[i]}: {e}")
            res = {}

        tone_keys = ['anger', 'disgust', 'fear', 'joy', 'sadness']
        tones_list = []
        for tone in tone_keys:
            score = res.get(tone, 0.0)
            try:
                score = float(score)
            except:
                score = 0.0
            tones_list.append((tone.capitalize(), round(score * 100, 2)))

        tone_results.append(tones_list)

    # Prepare tone values
    anger = [tone_results[i][0][1] for i in range(3)]
    disgust = [tone_results[i][1][1] for i in range(3)]
    fear = [tone_results[i][2][1] for i in range(3)]
    joy = [tone_results[i][3][1] for i in range(3)]
    sadness = [tone_results[i][4][1] for i in range(3)]

    # Plot bar chart
    values = np.arange(3)
    width = 0.15
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    plt.bar(values - 2*width, anger, width, label='Anger')
    plt.bar(values - width, disgust, width, label='Disgust')
    plt.bar(values, fear, width, label='Fear')
    plt.bar(values + width, joy, width, label='Joy')
    plt.bar(values + 2*width, sadness, width, label='Sadness')

    plt.xticks(values, ['Question 1', 'Question 2', 'Question 3'], fontsize=12, fontweight='bold')
    plt.yticks(fontsize=10)
    plt.legend(loc='upper right', fontsize='medium')
    plt.tight_layout()
    plt.savefig(f'./static/{session_id}_tone_analysis.jpg')
    plt.close()

    # Save text answers
    with open(f'./static/{session_id}_answers.json', 'w') as f:
        json.dump(responses, f, indent=4)

    # Video processing
    combined_path = f"./static/{session_id}_combined.avi"
    size = (1280, 720)
    fps = 20

    video = cv2.VideoWriter(combined_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    for p in paths:
        cap = cv2.VideoCapture(p)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, size)
            video.write(frame_resized)
        cap.release()
    video.release()

    # FER Analysis
    try:
        detector = FER(mtcnn=True)
        input_vid = Video(combined_path)
        data = input_vid.analyze(detector, display=False, save_frames=False, save_video=False,
                                 annotate_frames=False, zip_images=False)
        df = input_vid.to_pandas(data)
        df = input_vid.get_first_face(df)
        df = input_vid.get_emotions(df)

        pltfig = df.plot(figsize=(12, 6), fontsize=12).get_figure()
        plt.legend(fontsize='medium', loc='upper right')
        pltfig.savefig(f'./static/{session_id}_fer_output.png')
        plt.close()
    except Exception as e:
        print(f"[ERROR] FER analysis failed: {e}")

    return jsonify({"status": "success"})


# Interview completed response message
@app.route('/recorded')
def response():
    return render_template('recorded.html')


# Display results to interviewee
@app.route('/info')
def info():
    email=request.args.get('email')
    if not email:
        return "Email parameter is required",400
    cursor=user_db.connection.cursor()
    cursor.execute("SELECT session_id FROM candidates WHERE email=%s",(email,))
    row=cursor.fetchone()
    if not row:
        return "User not found",404
    session_id=row[0]
    try:
        with open(f'./static/{session_id}_result.json' , 'r') as file:
            output = json.load(file)
        with open(f'./static/{session_id}_answers.json' , 'r') as file:
            answers = json.load(file)
    except FileNotFoundError:
        return "resulkt not found",404



    return render_template('result.html' , output = output , responses = answers,timestamp=datetime.now().timestamp(), session_id=session_id)


# Send job confirmation mail to selected candidate
@app.route('/accept', methods=['GET'])
def accept():
    session_id = request.args.get('session_id')
    if not session_id:
        return "Session ID is required", 400

    result_path = f'./static/{session_id}_result.json'
    if not os.path.exists(result_path):
        return "Result file not found", 404

    with open(result_path, 'r') as file:
        output = json.load(file)

    name = output.get('Name', 'Candidate')
    email = output.get('Email')
    if not email:
        return "Candidate email not found", 500

    position = "Software Development Engineer"

    msg = Message(
        subject='Job Confirmation Letter',
        sender=MAIL_USERNAME,
        recipients=[email]
    )
    msg.body = f"""Dear {name},

Thank you for taking the time to interview for the {position} position. We enjoyed getting to know you.

I am pleased to inform you that we would like to offer you the {position} position. Your starting salary will be $15,000 per year with an anticipated start date of July 1.

Please respond to this email by June 23 to let us know if you would like to accept the position.

Sincerely,  
Vidushi Baliyan 
Human Resources Director"""

    try:
        mail.send(msg)
        print(f"[EMAIL SENT] Confirmation sent to {email}")
        return "Email sent successfully."
    except Exception as e:
        print(f"[SMTP ERROR] {e}")
        return f"Failed to send email: {e}", 500


# Send rejection mail
@app.route('/reject', methods=['GET'])
def reject():
    session_id = request.args.get('session_id')
    if not session_id:
        return "Session ID is required", 400

    result_path = f'./static/{session_id}_result.json'
    if not os.path.exists(result_path):
        return "Result file not found", 404

    with open(result_path, 'r') as file:
        output = json.load(file)

    name = output['Name']
    email = output['Email']
    position = "Software Development Engineer"

    msg = Message('Your application to Smart Hire', sender=MAIL_USERNAME, recipients=[email])
    msg.body = f"""Dear {name},

Thank you for considering Smart Hire. We’ve chosen to move forward with a different candidate for the {position} position.

We appreciate your time and encourage you to apply again in the future.

Regards,
Vidushi Baliyan
Human Resources Director"""
    try:
        mail.send(msg)
    except Exception as e:
        print(f"[SMTP ERROR] {e}")
        return f"Failed to send email: {e}", 500

    return "success"




if __name__ == '__main__':
    app.run(debug = True)