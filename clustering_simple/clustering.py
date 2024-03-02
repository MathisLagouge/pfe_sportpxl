
# Imports
import random, os, shutil
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from extract_obj import get_img_from_dir
import cv2


class image_clustering:

	def __init__(self, folder_path="data", n_clusters=10, max_examples=None):
		paths = os.listdir(folder_path)
		if max_examples == None:
			self.max_examples = len(paths)
		else:
			if max_examples > len(paths):
				self.max_examples = len(paths)
			else:
				self.max_examples = max_examples
		self.n_clusters = n_clusters
		self.folder_path = folder_path
		random.shuffle(paths)
		self.image_paths = paths[:self.max_examples]
		del paths 
		try:
			shutil.rmtree("output")
		except FileExistsError:
			pass
		print("\n output folders created.")
		os.makedirs("output")
		for i in range(self.n_clusters):
			os.makedirs("output/cluster" + str(i))
		print("\n Object of class \"image_clustering\" has been initialized.")

	def load_images(self):
		self.images = []
		for image in self.image_paths:
			self.images.append(cv2.cvtColor(cv2.resize(cv2.imread(self.folder_path + "/" + image), (1024,1024)), cv2.COLOR_BGR2RGB))
		self.images = np.float32(self.images).reshape(len(self.images), -1)
		self.images /= 255
		print("\n " + str(self.max_examples) + " images from the \"" + self.folder_path + "\" folder have been loaded in a random order.")

	def clustering(self):
		model = KMeans(n_clusters=self.n_clusters, random_state=728)
		model.fit(self.images)
		predictions = model.predict(self.images)
		#print(predictions)
		for i in range(self.max_examples):
			shutil.copyfile(self.folder_path+"/"+self.image_paths[i], "./output/cluster"+str(predictions[i])+"/"+self.image_paths[i])
		print("\n Clustering complete! \n\n Clusters and the respective images are stored in the \"output\" folder.")


def clustering(directory, number_of_clusters):

	print("\n\n \t\t START\n\n")

	temp = image_clustering(directory, number_of_clusters)
	temp.load_images()
	temp.clustering()

	print("\n\n\t\t END\n\n")

def test(n_clusters, tags_path):

	print("\n\n \t\t TEST\n\n")

	#get tags
	tags = pd.read_csv(tags_path, sep=";")
	
	x = 0 #bien classé	
	y = 0 #mal classé
	for i in range(n_clusters):
		directory = "./output/cluster" + str(i)
		images = get_img_from_dir(directory)

		#get ids of images from 1 cluster
		ids = []
		for img in images:
			ids.append(tags.loc[tags['key'] == str(img), 'bibs'].values[0])
		
		#cluster become the id majority
		freq = {}
		for i in ids:
			freq[i] = freq.setdefault(i, 0) + 1
		print(freq)
		max_occ = max(freq.values())
		cluster_id = [k for k, v in freq.items() if v == max_occ][0]

		#with this cluster id, find good predict and not good predict

		for id in ids:
			if id == cluster_id:
				x += 1
			else:
				y += 1
	
	acc = x / (x+y)

	print("#####")
	print("")
	print("bien classé : " + str(x))
	print("mal classé : " + str(y))
	print("precision : " + str(acc))
	print("")
	print("#####")
	print("\n\n\t\t END\n\n")