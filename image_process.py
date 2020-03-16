import cv2
import os
from shutil import copyfile
import numpy as np
import scipy.io
import math 

ROOT = os.path.dirname(os.path.abspath(__file__))
process_path = os.path.join(ROOT, "on_process")

def get_max_id(folder_name):
	if os.path.exists(os.path.join(process_path, "curr_id", folder_name + ".txt")):
		f = open(os.path.join(process_path, "curr_id", folder_name + ".txt"),'r')
		max_id = f.read()
		f.close()
	else:
		max_id = 0
	return int(max_id)

def update_max_id(folder_name, max_id):
	f = open(os.path.join(process_path, "curr_id", folder_name + ".txt"),'w')
	f.write(str(max_id))
	f.close()

def add_point(file_name, point, folder_name): 
	max_id = get_max_id(folder_name)
	new_point = np.array([point[0],point[1],max_id + 1])
	update_max_id(folder_name, max_id + 1)
	matfile_path = os.path.join(ROOT,"data","mat_file", folder_name)
	if os.path.exists(os.path.join(matfile_path, file_name + ".mat")):
		data = scipy.io.loadmat(os.path.join(matfile_path, file_name + ".mat"))['annPoints']	
		if (data.size > 0):
			data = np.vstack([data, new_point])
		else:
			data = new_point
	else:
		data = new_point
	save_mathfile(file_name, data, folder_name)
	img = cv2.imread(os.path.join(process_path, "img", file_name))
	cv2.circle(img, (point[0], point[1]), 6, (0, 0, 255), -1) # Point color is red
	cv2.imwrite(os.path.join(ROOT,"on_process", "img", file_name),img)

def refill_point(file_name, point, folder_name):
	img_orginal = cv2.imread(os.path.join(ROOT,"data", "img", folder_name, file_name))
	img = cv2.imread(os.path.join(process_path, "img", file_name))
	for i in range(point[1]-8, point[1]+9):
		for j in range(point[0]-8,point[0]+9):
			img[i][j]=img_orginal[i][j]
	#Redraw points if they occlusion
	matfile_path = os.path.join(ROOT,"data","mat_file", folder_name)
	data = scipy.io.loadmat(os.path.join(matfile_path, file_name + ".mat"))['annPoints']
	for point_ in data:
		dist = math.sqrt((point[0]-point_[0])**2+(point[1]-point_[1])**2)
		if (dist<12):
			color = img[point_[0]][point_[1]]
			cv2.circle(img, (point_[0], point_[1]), 6, (int(color[0]), int(color[1]), int(color[2])), -1) #Same color
	cv2.imwrite(os.path.join(ROOT,"on_process", "img", file_name),img)

def un_select_point_pre(pre_file_name, file_name, point, folder_name):
	matfile_path = os.path.join(ROOT,"data","mat_file", folder_name)
	pre_data = scipy.io.loadmat(os.path.join(matfile_path, pre_file_name + ".mat"))['annPoints']
	pre_id_, pre_index_ = find_id(pre_data, point)
	pre_img = cv2.imread(os.path.join(process_path, "img", pre_file_name))
	#Find in curr frame
	data = scipy.io.loadmat(os.path.join(matfile_path, file_name + ".mat"))['annPoints']
	index_ = find_point(data, pre_id_)
	if (index_!=-1):
		cv2.circle(pre_img, (point[0], point[1]), 6, (255, 0, 0), -1) # Point color is blue
	else:
		cv2.circle(pre_img, (point[0], point[1]), 6, (0, 0, 255), -1) # Point color is red
	cv2.imwrite(os.path.join(process_path, "img", pre_file_name), pre_img)

def un_select_point_curr(pre_file_name, file_name, point, folder_name):
	if (point[2]==0):
		return
	matfile_path = os.path.join(ROOT,"data","mat_file", folder_name)
	#Curr frame
	data = scipy.io.loadmat(os.path.join(matfile_path, file_name + ".mat"))['annPoints']
	id_, index_ = find_id(data, point)
	img = cv2.imread(os.path.join(process_path, "img", file_name))
	if (os.path.exists(os.path.join(matfile_path, pre_file_name + ".mat"))):
		pre_data = scipy.io.loadmat(os.path.join(matfile_path, pre_file_name + ".mat"))['annPoints']
	else:
		pre_data = []
	pre_index_ = find_point(pre_data, id_)
	if (pre_index_!=-1):
		cv2.circle(img, (point[0], point[1]), 6, (255, 0, 0), -1) # Point color is blue
		#Pre frame
		pre_img = cv2.imread(os.path.join(process_path, "img", pre_file_name))
		cv2.circle(pre_img, (pre_data[pre_index_][0], pre_data[pre_index_][1]), 6, (255, 0, 0), -1) # Point color is blue
		cv2.imwrite(os.path.join(process_path, "img", pre_file_name), pre_img)
	else:
		cv2.circle(img, (point[0], point[1]), 6, (0, 0, 255), -1) # Point color is red
	cv2.imwrite(os.path.join(process_path, "img", file_name), img)
	
def select_point_curr(old_select_point, pre_file_name, file_name, point, folder_name): 
	matfile_path = os.path.join(ROOT,"data","mat_file", folder_name)
	#Un green old select point
	un_select_point_curr(pre_file_name, file_name, old_select_point, folder_name)	
	#Open img and color green select point
	img = cv2.imread(os.path.join(process_path, "img", file_name))
	cv2.circle(img, (point[0], point[1]), 6, (0, 255, 0), -1) # Point color is green
	cv2.imwrite(os.path.join(process_path, "img", file_name),img)
	#Match with same id point in re-frame
	if (os.path.exists(os.path.join(matfile_path, pre_file_name + ".mat"))):
		pre_data = scipy.io.loadmat(os.path.join(matfile_path, pre_file_name + ".mat"))['annPoints']
	else:
		pre_data = []
	pre_index_ = find_point(pre_data, point[2])
	if (pre_index_!=-1):
		pre_img = cv2.imread(os.path.join(process_path, "img", pre_file_name))
		cv2.circle(pre_img, (pre_data[pre_index_][0], pre_data[pre_index_][1]), 6, (0, 255, 0), -1) # Point color is green
		cv2.imwrite(os.path.join(process_path, "img", pre_file_name),pre_img)

def select_point_pre(pre_file_name, point):
	img = cv2.imread(os.path.join(process_path, "img", pre_file_name))
	cv2.circle(img, (point[0], point[1]), 6, (0, 255, 0), -1) # Point color is green
	cv2.imwrite(os.path.join(process_path, "img", pre_file_name),img)	

def select_point(file_name, point, folder_name): #Select nearest point which is in annPoints in curr frame
	matfile_path = os.path.join(ROOT, "data", "mat_file", folder_name)
	data = scipy.io.loadmat(os.path.join(matfile_path, file_name + ".mat"))['annPoints']
	select_point = np.array([-1,-1,0])
	min_dist = 6.001
	for point_ in data:
		dist = math.sqrt((point[0]-point_[0])**2+(point[1]-point_[1])**2)
		if dist < min_dist:
			min_dist = dist
			select_point = point_
	return select_point

def find_id(data, point):
	id_ = 0 
	index = -1
	for index_, point_ in enumerate(data):
		if(point_[0]==point[0] and point_[1]==point[1]):
			id_ = point_[2]
			index = index_
			break	
	return id_, index

def find_point(data, id_): #Find point while know it's id
	index = -1
	for index_, point_ in enumerate(data):
		if (point_[2] == id_):
			index = index_
			break
	return index

def delete_id(file_name, folder_name, id_, num):
	matfile_path = os.path.join(ROOT, "data", "mat_file", folder_name)
	max_id = get_max_id(folder_name)
	file_index = int(file_name[:file_name.find(".")]) + num
	check_pre_frame = file_index
	extension = file_name[file_name.find("."):]
	file_name_ = str(file_index) + extension
	while (os.path.exists(os.path.join(matfile_path, file_name_  + ".mat"))):
		data = scipy.io.loadmat(os.path.join(matfile_path, file_name_  + ".mat"))['annPoints']
		index = find_point(data, id_)
		if (index!=-1):
			data[index][2] = max_id
			max_id += 1
			save_mathfile(file_name_, data, folder_name)
		else:
			break
		file_index += num
		file_name_ = str(file_index) + extension
	update_max_id(folder_name, max_id)

def delete_point(pre_file_name, file_name, point, folder_name, type_):
	matfile_path = os.path.join(ROOT, "data", "mat_file", folder_name)
	#Find id of point
	data = scipy.io.loadmat(os.path.join(matfile_path, file_name + ".mat"))['annPoints']
	id_, index_ = find_id(data, point)
	#Refill point in on_process img and delete point in current file
	#refill_point(file_name, point, folder_name)
	data = np.delete(data, (index_), axis = 0)
	save_mathfile(file_name, data, folder_name)
	if os.path.exists(os.path.join(matfile_path, pre_file_name + ".mat")):
		pre_file_mat = scipy.io.loadmat(os.path.join(matfile_path, pre_file_name + ".mat"))
		pre_data = pre_file_mat['annPoints']
		index_ = find_point(pre_data, id_)
		#Unselect point in pre_frame if it has match
		if (index_!=-1):
			pre_img = cv2.imread(os.path.join(process_path, "img", pre_file_name))
			cv2.circle(pre_img, (pre_data[index_][0], pre_data[index_][1]), 6, (0, 0, 255), -1) # Point color is red
			cv2.imwrite(os.path.join(process_path, "img", pre_file_name), pre_img)	
	else: 
		pre_data = []
	load_img_annPoints(file_name, data, pre_data, folder_name)
	#delete id in after/before files
	if (id_!=0):
		if (type_ == "after"):
			delete_id(file_name, folder_name, id_, 1)
		elif (type_ == "before"):
			delete_id(file_name, folder_name, id_, -1)

def delete_point_after(pre_file_name, file_name, point, folder_name):
	delete_point(pre_file_name, file_name, point, folder_name, "after")

def delete_point_before(pre_file_name, file_name, point, folder_name):
	delete_point(pre_file_name, file_name, point, folder_name, "before")

def match(pre_file_name, pre_point, file_name, point, folder_name):
	matfile_path = os.path.join(ROOT, "data", "mat_file", folder_name)
	img = cv2.imread(os.path.join(process_path, "img",  file_name))
	pre_img = cv2.imread(os.path.join(process_path, "img", pre_file_name))
	#Load pre file mat to get id of point
	pre_data = scipy.io.loadmat(os.path.join(matfile_path, pre_file_name + ".mat"))['annPoints']
	id_, pre_index_ = find_id(pre_data, pre_point)
	#Process curr frame data
	data = scipy.io.loadmat(os.path.join(matfile_path, file_name + ".mat"))['annPoints']
	curr_id_, index_ = find_id(data, point)
	old_pre_index = find_point(pre_data, point[2])
	if (old_pre_index!=-1):	
		index_old_pre_match_curr = find_point(data, pre_data[pre_index_][2])
		if (index_old_pre_match_curr!=-1):
			cv2.circle(pre_img, (pre_data[old_pre_index][0], pre_data[old_pre_index][1]), 6, (255, 0, 0), -1) # Point color is blue
			data[index_old_pre_match_curr][2]=pre_data[old_pre_index][2] #Swap it
		else:  
			cv2.circle(pre_img, (pre_data[old_pre_index][0], pre_data[old_pre_index][1]), 6, (0, 0, 255), -1) # Point color is red
	data[index_][2] = id_
	cv2.circle(pre_img, (pre_point[0], pre_point[1]), 6, (255, 0, 0), -1) # Point color is blue
	cv2.imwrite(os.path.join(process_path, "img", pre_file_name),pre_img)	
	cv2.circle(img, (point[0], point[1]), 6, (255, 0, 0), -1) # Point color is blue
	cv2.imwrite(os.path.join(process_path, "img", file_name),img)	
	save_mathfile(file_name, data, folder_name)

def move(pre_file_name, file_name, curr_point, move_point, folder_name):
	matfile_path = os.path.join(ROOT,"data","mat_file", folder_name)
	data = scipy.io.loadmat(os.path.join(matfile_path, file_name + ".mat"))['annPoints']
	id_, index_ = find_id(data, curr_point)
	data[index_][0] = move_point[0]
	data[index_][1] = move_point[1]
	save_mathfile(file_name, data, folder_name)
	# img = cv2.imread(os.path.join(process_path, "img", file_name))
	# color = img[curr_point[0]][curr_point[1]]
	# refill_point(file_name, curr_point, folder_name)
	# cv2.circle(img, (point[0], point[1]), 6, (int(color[0]), int(color[1]), int(color[2])), -1) # Point color is same color curr_point
	# cv2.imwrite(os.path.join(process_path, "img", file_name),img)
	if os.path.exists(os.path.join(matfile_path, pre_file_name + ".mat")):
		pre_file_mat = scipy.io.loadmat(os.path.join(matfile_path, pre_file_name + ".mat"))
		pre_data = pre_file_mat['annPoints']
	else: 
		pre_data = []
	load_img_annPoints(file_name, data, pre_data, folder_name)
	load_img_annPoints(pre_file_name, pre_data, data, folder_name)

def load_img_annPoints(file_name, data, pre_data, folder_name):
	img_path = os.path.join(process_path, "img", file_name)
	img = cv2.imread(os.path.join(ROOT, "data", "img", folder_name, file_name))
	for point in data:
		center_coordinates = (point[0], point[1])
		index_ = find_point(pre_data, point[2])
		if (index_==-1):
			cv2.circle(img, center_coordinates, 6, (0, 0, 255), -1) #Point color is red if it hasn't match
		else:
			cv2.circle(img, center_coordinates, 6, (255, 0, 0), -1) #Point color is blue if it has match
	cv2.imwrite(img_path,img)

def save_mathfile(file_name, data, folder_name):
	matfile_path = os.path.join(ROOT,"data","mat_file", folder_name, file_name + ".mat")
	mdict = {}
	mdict['__header__'] = b'MATLAB 5.0 MAT-file'
	mdict['__version__'] = 0x0100
	mdict['__globals__'] = []
	mdict['annPoints'] = np.array(data)
	scipy.io.savemat(matfile_path, mdict)

def load_matfile(pre_file_name, file_name, folder_name):
	matfile_path = os.path.join(ROOT,"data","mat_file", folder_name)

	file_name_mat = file_name + ".mat"
	if os.path.exists(os.path.join(matfile_path, file_name_mat)):
		file_mat = scipy.io.loadmat(os.path.join(matfile_path, file_name_mat))
		data = file_mat['annPoints']
	else:
		data = []

	pre_file_name_mat = pre_file_name + ".mat"
	if os.path.exists(os.path.join(matfile_path, pre_file_name_mat)):
		pre_file_mat = scipy.io.loadmat(os.path.join(matfile_path, pre_file_name_mat))
		pre_data = pre_file_mat['annPoints']
	else: 
		pre_data = []

	load_img_annPoints(file_name, data, pre_data, folder_name)
	load_img_annPoints(pre_file_name, pre_data, data, folder_name)

def load_img_on_process(pre_file_name, file_name, folder_name):
	folder_path = os.path.join(ROOT, "data", "img", folder_name)

	#Delete all file in on_process folder
	filelist = [ f for f in os.listdir(os.path.join(process_path, "img"))]
	for f in filelist:
		os.remove(os.path.join(process_path, "img", f))

    #Copy 2 process image to on_process folder
	#copyfile(os.path.join(folder_path, file_name), os.path.join(process_path, file_name))
	#copyfile(os.path.join(folder_path, pre_file_name), os.path.join(process_path, pre_file_name))
	load_matfile(pre_file_name, file_name, folder_name)