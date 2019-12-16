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
		# check for image key in payload
		if 'image' not in request.data:
			return Response({'error':'Please upload a file.'},status=status.HTTP_400_BAD_REQUEST)
		# check extension of the file
		if str.endswith(request.data['image'].name, '.jpeg' ) or \
		str.endswith(request.data['image'].name, '.jpg' ) or \
		str.endswith(request.data['image'].name, '.png' ):
			img = image.img_to_array(image.load_img(request.data['image'], target_size=(80, 80), grayscale=True))
		else:
			return Response({'error':'Please upload correct file ending with png or jpeg.'},status=status.HTTP_400_BAD_REQUEST)

		# preprocess the image
		# img = np.array(img)
		img = img.astype('float64')

		# Creating payload for TensorFlow serving request
		payload = {
		    "instances": [{'input_image': img.tolist()}]
		}
		# Making POST request
		r = requests.post('http://localhost:9000/v1/models/classifier:predict', json=payload)

		# Decoding results from TensorFlow Serving server
		pred = json.loads(r.content.decode('utf-8'))
		return Response(pred, status=status.HTTP_200_OK)