from flask_wtf import FlaskForm, RecaptchaField 
from wtforms import StringField, SelectField, ValidationError , PasswordField ,SubmitField , validators
from wtforms.validators import DataRequired, Length, Email , EqualTo
from MongoPackageV2 import *



class SignUpForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    Email = StringField('Email', validators=[DataRequired(), Email()])
    phoneNumber = StringField('Phone Number', validators=[DataRequired() , Length(11)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, max=20)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password', message='Passwords must match'), Length(min=6, max=20)])
    submit = SubmitField('Sign Up')



class SignInForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Sign In')



class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Submit')


#####################################################

class applyModel(FlaskForm):

   
    
    x = finding_camera_names()
    cameraName = SelectField('Camera Name', choices = x ,validators=[DataRequired()])
    models = ['' , 'violence' , 'vehicle' , 'crowdedDensity' , 'crossingBorder' , 'crowded' ,'Gender']
    Model = SelectField('Model' , choices = models , validators=[DataRequired()] )


########################################################

class searchInDB(FlaskForm):
    
    days =[''] + list(range(1, 32))
    months =[''] + list(range(1, 13))
    years = ['' , 2024]
    cameraNames =  [''] + finding_camera_names()
    modelsNames = ['' , 'violence' , 'vehicle' , 'crowdedDensity' , 'crossingBorder' , 'crowded' , 'Gender']
    
        

    day_selector = SelectField('Day', choices=days ,validators=[DataRequired()])
    month_selector = SelectField('Month', choices=months,validators=[DataRequired()])
    year_selector = SelectField('Year', choices=years,validators=[DataRequired()])
    cameraName = SelectField('Camera Name', choices=cameraNames , validators=[DataRequired()] ) 
    modelName = SelectField('Model Name', choices=modelsNames , validators=[DataRequired()] ) 

    
########################################################
class AddCamera(FlaskForm):

    cameraName = StringField('CameraName',validators=[DataRequired()])
    sources = ['' ,'WEBCAM' , 'RTSP' , 'URL']
    sourceType = SelectField('SourceType' , choices= sources ,validators=[DataRequired()])
    source = StringField('Source', validators=[DataRequired()])
    location = StringField('Location', validators=[DataRequired()] ,render_kw={'readonly': True})
