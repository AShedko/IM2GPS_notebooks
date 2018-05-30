from bottle import route, run, get, post, request
from predict import *
import PIL
import urllib

@route('/req', method= "GET")
def index():
	res = urllib.request.urlretrieve(request.params["url"], "test.jpg")
	im = PIL.Image.open("test.jpg")
	return {'result': str(predict(im))}

run(host='localhost', port=5533)
