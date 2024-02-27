# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     form = SignUpForm()
#     if form.validate_on_submit():
#         users_collection = db['users']
#         new_user = {
#             'Username': form.username.data,
#             'Email': form.Email.data,
#             'Phone': form.phoneNumber.data,
#             'Password': form.password.data
#         }
#         users_collection.insert_one(new_user)
#         flash('Account created successfully!', 'success')
#         return redirect(url_for('signin'))
#     return render_template('signup.html', form=form)


# @app.route('/signin', methods=['GET', 'POST'])
# def signin():
#     form = SignInForm()
#     if form.validate_on_submit():
#         username = form.username.data
#         password = form.password.data
#         user = db['users'].find_one({'Username': username, 'Password': password})
#         if user:
#             user_obj = User()
#             user_obj.id = str(user['_id'])  
#             login_user(user_obj)
#             next_page = request.args.get('next')  
#             return redirect(next_page or url_for('apply_Model'))
#         else:
#             flash('Invalid username or password', 'error')
#     return render_template('signin.html', form=form)

