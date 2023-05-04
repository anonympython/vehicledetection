from flask import Flask, render_template, send_file
from datetime import datetime
from fileinput import filename
from flask import *  
from track import *
from live_tracker import *
from live_annotate import *

app = Flask(__name__)


from flask_sqlalchemy import SQLAlchemy
 
# Initialize flask  and create sqlite database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
 
# create datatable
class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(50))
    data = db.Column(db.LargeBinary)

@app.route('/')
def index_page_ren():
    return render_template("index.html")

@app.route('/login', methods =['GET', 'POST'])
def login_page_ren():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        if username=="Aryan" and password=="user@123":
            msg = username
            return render_template('upload.html', msg = msg)
        else:
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg = msg)

@app.route('/upload')  
def upload():  
    return render_template("upload.html")  
  
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        type_of_detection = request.form['seg_type']
        f.save(f.filename)
        upload = Upload(filename=f.filename, data=f.read())
        segment = 0
        if(type_of_detection == "segment"):
            segment=1
        data,temp = process(f.filename,segment)
        if(segment==0):  
            path = "Detected_processed_video_" + f.filename 
        else:
            path = "Segmentated_processed_video_" + f.filename 
        vehicle_count,vehicle = get_count(data)
        return redirect(url_for("download_file",path = path,vehicle_count = vehicle_count, vehicle =  vehicle))

@app.route('/download')
def download_file():
    path = request.args['path']
    vehicle_count = request.args['vehicle_count']
    vehicle = request.args['vehicle']
    print(path)
    return render_template("download.html",file=path,vehicle=vehicle,vehicle_count=vehicle_count)

@app.route('/livecam')
def livecam():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/getfile/<path:filename>', methods=['GET'])
def getfile(filename):
    print(filename)
    return send_file(filename , as_attachment=True)

@app.route('/livecam')
def test():
    return Response(time = datetime.datetime.now())

@app.route('/tasks')  
def tasks():  
    return render_template("tasks.html") 

@app.route('/viewprocessed/<path:filename>', methods=['GET'])
def viewprocessed(filename):
    return render_template("processed_video.html",video_path=filename) 

if __name__ == "__main__":
    app.run(debug=True)