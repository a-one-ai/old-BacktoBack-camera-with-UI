from flask import Flask , render_template , request , redirect , url_for, jsonify , flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import g
from form import *
from takeOneFrame import *           
from MongoPackageV2 import *
from threading import Thread
import requests
import os 
from flask_mail import Mail, Message 
import random
import smtplib
from werkzeug.security import check_password_hash , generate_password_hash
from bson.objectid import ObjectId
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
#num_cpu_cores = torch.multiprocessing.cpu_count()
#torch.set_num_threads(4)  # Set the number of threads for PyTorch


# Set GPU device if available
if torch.cuda.is_available():
    print('gpu')
    torch.cuda.set_device(device)
else :
        print('no gpu')

num_cpu_cores = torch.multiprocessing.cpu_count()


app = Flask(__name__ )
#____________________________________________________________
app.config['SECRET_KEY'] = os.urandom(32)


mail = Mail(app)
#____________________________________________________________

@app.route('/')
def hello_world():
    return render_template('base.html')
#____________________________________________________________

login_manager = LoginManager(app)
login_manager.login_view = 'signin'

class User(UserMixin):
    pass

@login_manager.user_loader
def load_user(user_id):
    user = User()
    user.id = user_id
    return user

@app.before_request
def before_request():
    if not current_user.is_authenticated and request.endpoint and request.endpoint not in ['signin', 'static']:
        g.return_url = request.endpoint
        return redirect(url_for('signin'))



#____________________________________________________________
    
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignUpForm()
    if form.validate_on_submit():
        users_collection = db['users']
        hashed_password = generate_password_hash(form.password.data)
        new_user = {
            'Username': form.username.data,
            'Email': form.Email.data,
            'Phone': form.phoneNumber.data,
            'Password': hashed_password  
        }
        users_collection.insert_one(new_user)
        flash('Account created successfully!', 'success')
        return redirect(url_for('signin'))
    return render_template('signup.html', form=form)





@app.route('/signin', methods=['GET', 'POST'])
def signin():
    form = SignInForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        
        user = db['users'].find_one({'Username': username})
        if user and check_password_hash(user['Password'], password):
            user_obj = User()
            user_obj.id = str(user['_id']) 
            login_user(user_obj)
            next_page = request.args.get('next') 
            return redirect(next_page or url_for('apply_Model'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('signin.html', form=form)




@app.route('/logout')
@login_required
def logout():
    logout_user()
    form = SignInForm()
    return render_template('signin.html', form=form)



#____________________________________________________________

# def send_email(header, body, recipient_email):
#     sender_email = 'shaimaaartificial@gmail.com'
#     sender_password = 'ekle eqjb jnkm rztm'
#     message = f"{header}\n\n{body}"

#     with smtplib.SMTP('smtp.gmail.com', 587) as server:
#         server.starttls()
#         server.login(sender_email, sender_password)
#         server.sendmail(sender_email, recipient_email, message)




# @app.route('/forgot_password', methods=['GET', 'POST'])
# def forgot_password():
#     fForm = ForgotPasswordForm()
#     if fForm.validate_on_submit():
#         email = fForm.email.data
#         user = db['users'].find_one({'email': email})
        
#         if user:
#             code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
#             db['users'].update_one({'email': email}, {'$set': {'reset_code': code}})
#             recipients = email
#             header = 'Password Reset Code'
#             body = f'Your password reset code is: {code}'
#             send_email(header, body, recipients)

#             app.logger.info("Password reset code sent successfully.")
            
#             return redirect(url_for('confirm_reset_code'))
#         else:
#             app.logger.error("Invalid email provided for password reset.")

#     return render_template('forgot_password.html', form=fForm)





import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(header, body, recipient_email):
    sender_email = 'shaimaaartificial@gmail.com'
    sender_password = 'ekle eqjb jnkm rztm'
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = header

    message.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = message.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")



@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    fForm = ForgotPasswordForm()
    if fForm.validate_on_submit():
        email = fForm.email.data
        user = db['users'].find_one({'Email': email})
        
        if user:
            code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
            db['users'].update_one({'Email': email}, {'$set': {'reset_code': code}})
            recipients = email
            header = 'Password Reset Code'
            body = f'Your password reset code is: {code}'
            send_email(header, body, recipients)

            app.logger.info("Password reset code sent successfully.")
            
            return redirect(url_for('confirm_reset_code'))
        else:
            app.logger.error("Invalid email provided for password reset.")

    return render_template('forgot_password.html', form=fForm)




@app.route('/confirm_reset_code', methods=['GET', 'POST'])
def confirm_reset_code():
    if request.method == 'POST':
        email = request.form['email']
        code = request.form['code']
        user = db['users'].find_one({'email': email, 'reset_code': code})
        if user:
            return redirect(url_for('reset_password', user_id=user['_id']))
        else:
            flash('Invalid email or code.', 'error')
    return render_template('confirm_reset_code.html')




@app.route('/reset_password/<user_id>', methods=['GET', 'POST'])
def reset_password(user_id):
    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        if new_password == confirm_password:
            hashed_password = generate_password_hash(new_password)  # Hash the new password
            db['users'].update_one({'_id': ObjectId(user_id)}, {'$set': {'Password': hashed_password, 'reset_code': None}})
            flash('Password has been reset successfully.', 'success')
            return redirect(url_for('signin'))  
        else:
            flash('Passwords do not match.', 'error')
    return render_template('reset_password.html')




#____________________________________________________________
@app.route('/addCamera' ,  methods=['GET', 'POST'])
@login_required
def addCamera():
    addCameraForm = AddCamera()
    if addCameraForm.validate_on_submit():
        # finding_camera_names()
        cameraName = addCameraForm.cameraName.data
        sourceType = addCameraForm.sourceType.data
        source = addCameraForm.source.data
        x, y = map(int, addCameraForm.location.data[1:-1].split(','))
        

        insert_camera_info(cameraName, sourceType, source, x , y)
        
        print(cameraName, sourceType, source, x , y)
        return render_template('cameraAdded.html', title='addCamera', form=addCameraForm )

        
        
    return render_template('addCamera.html', title='addCamera', form=addCameraForm)


#____________________________________________________________

@app.route('/applyModel', methods=['GET', 'POST'])
@login_required
def apply_Model():
    form = applyModel()
    if form.validate_on_submit():
        # Get form data
        cameraName = form.cameraName.data
        Model = form.Model.data


        # Include form data in the redirect URL
        return redirect(url_for('running', cameraName=cameraName,  Model=Model , form=form))
    
    # If the form is not submitted or not valid, render the form template
    return render_template('form.html', title='Get Data', form=form)
#____________________________________________________________
@app.route('/running', methods=['GET', 'POST'])
@login_required
def running():

    cameraName = request.args.get('cameraName')
    modelName = request.args.get('Model')
    result = processInsert(cameraName, modelName)
    return render_template('model_running.html', title='modelRunning', result= result)

#____________________________________________________________
@app.route('/search' , methods = ['GET', 'POST'])
@login_required
def search():
    searchForm = searchInDB()
    result = None 
    if searchForm.validate_on_submit():
        
        cameraName = searchForm.cameraName.data
        modelName = searchForm.modelName.data
        daySelector = searchForm.day_selector.data 
        monthSelector = searchForm.month_selector.data
        yearSelector = searchForm.year_selector.data

        result = date_filter_aggerigates_df(cameraName , modelName ,daySelector, monthSelector, yearSelector)



    return render_template('search.html', title='seach in database', form=searchForm , result=result)


#____________________________________________________________
collection = db['CameraInfo']
def watch_changes():
    change_stream = collection.watch(full_document='updateLookup') 
    for change in change_stream:
        if change['operationType'] == 'insert':
            print("New Camera Name Inserted:", change['fullDocument']['Camera Name'])
        elif change['operationType'] == 'delete':
            print("Document Deleted:", change['documentKey']['_id'])



@app.route('/camera_names'  ,methods=['GET', 'POST'])
def get_camera_names():
    cursor = collection.find({})
    camera_names_list = [document['Camera Name'] for document in cursor]
    return jsonify(camera_names_list)
#____________________________________________________________

if __name__ == '__main__':
    # Start change stream watcher in a separate thread
    # change_stream_thread = Thread(target=watch_changes)
    # change_stream_thread.start()
    app.run(host='0.0.0.0',port=8080 , debug=True)

