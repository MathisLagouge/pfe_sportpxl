#main

#import
from extract_obj import extract_obj_from_images
from extract_obj import get_img_from_dir
from clustering import clustering

import timeit

DIRECTORY = "../datasets/YoungtimersSuperBerline/"
SAVE_CROP_DIRECTORY = "./sub/"
MODEL_PATH = "./best.pt"
SIZE = (900,900)  #img resize for clustering
N_CLUSTERS = 10

start = timeit.default_timer()
#get img test as list of openCV object
images = get_img_from_dir(DIRECTORY)
nb_get_img = len(images)
end = timeit.default_timer()
time1 = end - start

start = timeit.default_timer()
#get target obj from images as list of openCV object
#save target obj in ./img/sub/
#keep only picture with one cars tag
target_obj = extract_obj_from_images(images, SIZE, MODEL_PATH, saving_img=True, saving_img_path=SAVE_CROP_DIRECTORY)
del images
del target_obj
end = timeit.default_timer()
time2 = end - start

start = timeit.default_timer()
#clustering
#saving img in differents clusters in ./output
clustering(SAVE_CROP_DIRECTORY, N_CLUSTERS,)
end = timeit.default_timer()
time3 = end - start

print("")
print("###")
print("temps d'execution get image by course : " + str(time1))
print("nombre d'images récupérées : " + str(nb_get_img) + "   parmis 1598 images du datasets")
print("###")
print("temps d'execution extract object from img : " + str(time2))
print("###")
print("temps d'execution clustering : " + str(time3))
print("###")