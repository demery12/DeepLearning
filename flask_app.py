#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request, flash, send_from_directory
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
# prevents dangerous filenames
# (we don't actually store them, but maybe this will one day be needed)
from werkzeug.utils import secure_filename

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
import pickle
from PIL import Image
import text_boxer_app
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import * 
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()

with open('num_to_word.pickle', 'rb') as handle:
    num_to_word = pickle.load(handle)

#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(imgstr.decode('base64'))
	

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/tmp/<filename>')
def get_file(filename):
	return send_from_directory('/tmp/', filename)


# I'm following the guidelines of this:
# http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	##imgData = request.get_data()
	#encode it into a suitable format
	##convertImage(imgData)

	# This is not currently working, when no file is selected
	# the form still sends something, so this is never triggered
	if 'photo' not in request.files:
		flash("No file attached")
		return render_template("index.html")

	photo = request.files['photo']

	if photo.filename == '':
		flash('No file selected')
		return render_template("index.html")

	if photo and allowed_file(photo.filename):
		filename = secure_filename(photo.filename)
		photo.save('/tmp/' + filename)

	#read the image into memory
	#im = Image.open(photo)
	#im.show()
	#x = imread(photo,mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	#x = np.invert(x)
	#make it the right size
	words, filenames = text_boxer_app.box_image('/tmp/' + filename)
	response = []
	for word in words:
		im28 = word.resize((28, 28))
		x = np.array(im28, dtype=np.float32).flatten()
		#x = imresize(x,(28,28))
		#imshow(x)
		#convert to a 4D tensor to feed into our model
		x = x.reshape(1,28,28,1)
		print "debug2"
		#in our computation graph
		with graph.as_default():
			#perform the prediction
			out = model.predict(x, verbose=1)
			print(out)
			print(np.argmax(out,axis=1))
			print num_to_word[np.argmax(out,axis=1)[0]]
			print "debug3"
			#convert the response to a string
			response.append(num_to_word[np.argmax(out,axis=1)[0]])		
	return render_template('results.html', results = zip(response,filenames), words = words)	
	

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5005))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
#app.run(debug=True)
