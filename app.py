import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from image_process import *
import numpy as np

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

files = []
folder_name = []
old_action = "NULL"
old_select_point_curr_, old_select_point_pre_, select_point_curr_, select_point_pre_= [], [], [], []
bgPosX_, bgPosY_, bgWidth_, bgHeight_ = 0, 0, 500, 0

@app.route('/')
def home():
    return choose_folder()
#--------------------------------------------------------------------------------
@app.route('/choose_folder')
def choose_folder():
	return render_template('choose_folder.html')
#--------------------------------------------------------------------------------
@app.route('/upload/<folder_name>/<file_name>')
def send_image(folder_name, file_name):
	if (folder_name != 'NULL'):
		target = os.path.join(APP_ROOT, "data", "img", folder_name)
	else:
		target = os.path.join(APP_ROOT, "on_process", "img")
	return send_from_directory(target, file_name)
#--------------------------------------------------------------------------------
@app.route('/upload', methods = ["POST"])
def upload():
	global folder_name
	folder_name = str(request.files.getlist("files[]")[0])
	folder_name = folder_name[15:folder_name.find("/")]

	target = os.path.join(APP_ROOT, "data","img", folder_name)
	global files
	files = [f for f in os.listdir(target) if os.path.isfile(os.path.join(target, f))]

	return redirect(url_for('.annotate', folder_name = folder_name, file_name = files[1], action ='NULL'))
#--------------------------------------------------------------------------------
@app.route('/annotate/select/<folder_name>/<file_name>/<action>', methods = ["POST"])
def annotate_select(folder_name, file_name, action):
	global old_select_point_curr_, old_select_point_pre_, select_point_curr_, select_point_pre_
	global bgPosX_, bgPosY_, bgWidth_, bgHeight_
	x = request.form['x']
	y = request.form['y']
	status = request.form['status']
	bgPosX_ = request.form['bgPosX_']
	bgPosY_ = request.form['bgPosY_']
	bgWidth_ = request.form['bgWidth_']
	bgHeight_ = request.form['bgHeight_']
	action = request.form['action']
	type_submit = request.form['type_submit']

	pre_file_name = str(int(file_name[:file_name.find(".")]) - 1) + file_name[file_name.find("."):]
	if (type_submit == "select"): #Form submit is select point
		point = np.array([int(x), int(y), 0])
		if (status == "pre"):
			pre_file_name = str(int(file_name[:file_name.find(".")]) - 1) + file_name[file_name.find("."):]
			select_point_ = select_point(pre_file_name, point, folder_name)
			if (select_point_[2]!=0):
				un_select_point_pre(pre_file_name, file_name, old_select_point_pre_, folder_name)
				old_select_point_pre_ = select_point_pre_
				select_point_pre_ = select_point_
				select_point_pre(pre_file_name, select_point_pre_)
		else:
			if (action != 'ADD' and action != 'NULL' and action !="MOVE"):
				select_point_ = select_point(file_name, point, folder_name)
				if (select_point_[2]!=0):
					un_select_point_curr(pre_file_name, file_name, old_select_point_curr_, folder_name)
					old_select_point_curr_ = select_point_curr_
					select_point_curr_ = select_point_
					select_point_curr(old_select_point_curr_, pre_file_name, file_name, select_point_curr_, folder_name)
			elif (action == 'ADD'):
				add_point(file_name, point, folder_name)
				re_init()
			elif (action == 'MOVE'):
				if (select_point_curr_[2]==0):
					select_point_ = select_point(file_name, point, folder_name)
					if (select_point_[2]!=0):
						select_point_curr_ = select_point_
						select_point_curr(old_select_point_curr_, pre_file_name, file_name, select_point_curr_, folder_name)
				else:
					move(pre_file_name, file_name, select_point_curr_, point, folder_name)
					re_init()

	return redirect(url_for('.annotate', folder_name = folder_name, file_name = file_name, action = action))

#-------------------------------------------------------------------------------
def re_init():
	global old_select_point_curr_, old_select_point_pre_, select_point_curr_, select_point_pre_
	old_select_point_curr_ = np.array([-1,-1,0])
	old_select_point_pre_ = np.array([-1,-1,0])
	select_point_curr_ = np.array([-1,-1,0])
	select_point_pre_ = np.array([-1,-1,0])
#--------------------------------------------------------------------------------
@app.route('/annotate/action/<folder_name>/<file_name>/<action>')
def annotate_action(folder_name, file_name, action):
	global old_select_point_curr_, old_select_point_pre_, select_point_curr_, select_point_pre_
	global bgPosX_, bgPosY_, bgWidth_, bgHeight_
	pre_file_name = str(int(file_name[:file_name.find(".")]) - 1) + file_name[file_name.find("."):]
	if (action == "MATCH"):
		if (select_point_pre_[2]!=0 and select_point_curr_[2]!=0):
			pre_file_name = str(int(file_name[:file_name.find(".")]) - 1) + file_name[file_name.find("."):]
			match(pre_file_name, select_point_pre_, file_name, select_point_curr_, folder_name)
			re_init()
		else:
			return redirect(url_for('.annotate', folder_name = folder_name, file_name = file_name, action = action))
	elif (action == "DELETE_BEFORE"):
		if (select_point_curr_[2]!=0):
			delete_point_before(pre_file_name, file_name, select_point_curr_, folder_name)
			re_init()
		else:
			return redirect(url_for('.annotate', folder_name = folder_name, file_name = file_name, action = action))
	elif (action == "DELETE_AFTER"):
		if (select_point_curr_[2]!=0):
			delete_point_after(pre_file_name, file_name, select_point_curr_, folder_name)
			re_init()
		else:
			return redirect(url_for('.annotate', folder_name = folder_name, file_name = file_name, action = action))

	return redirect(url_for('.annotate', folder_name = folder_name, file_name = file_name, action = action))
#--------------------------------------------------------------------------------
@app.route('/annotate/<folder_name>/<file_name>/<action>')
def annotate(folder_name, file_name, action):
	pre_file_name = str(int(file_name[:file_name.find(".")]) - 1) + file_name[file_name.find("."):]
	if action == 'NULL':
		global bgPosX_, bgPosY_, bgWidth_, bgHeight_
		bgPosX_, bgPosY_, bgWidth_, bgHeight_ = 0, 0, 500, 0
		re_init()
		load_img_on_process(pre_file_name, file_name, folder_name)

	global old_action
	if (old_action!=action):
		old_action = action
		re_init()
		load_matfile(pre_file_name, file_name, folder_name)
	return render_template('annotate.html', files = files, folder_name = folder_name, file_name = file_name, pre_file_name = pre_file_name, action = action, bgPosX_ = bgPosX_, bgPosY_ = bgPosY_, bgWidth_ = bgWidth_, bgHeight_ = bgHeight_)
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)