from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from app.settings import MEDIA_ROOT
from django.core.files.storage import FileSystemStorage
import numpy as np
import requests
from keras.preprocessing import image
from io import BytesIO
import base64
import os
import json
# Create your views here.
class Classify(APIView):

	# predict class of image
	def post(self, request, format = None):
		#save image uploaded by user
		directory = os.path.join(MEDIA_ROOT, 'uploads')
		fs = FileSystemStorage()
		fs.save(directory + request.data['image'].name, request.data['image'])
		img = image.img_to_array(image.load_img(directory + request.data['image'].name, target_size=(80, 80), grayscale=True)) / 255.
		img = np.array(img)
		# this line is added because of a bug in tf_serving(1.10.0-dev)
		img = img.astype('float16')

		# Creating payload for TensorFlow serving request
		payload = {
		    "instances": [{'input_image': img.tolist()}]
		}

		# Making POST request
		r = requests.post('http://localhost:9000/v1/models/classifier:predict', json=payload)

		# Decoding results from TensorFlow Serving server
		pred = json.loads(r.content.decode('utf-8'))
		return Response(pred, status=status.HTTP_200_OK)
		return Response(status=status.HTTP_400_BAD_REQUEST)