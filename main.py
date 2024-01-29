#main

#import
import extract_obj as extract_obj

DIRECTORY = "./img/img_test"

#get img test as list of openCV object
images = extract_obj.get_img_test(DIRECTORY)

#get target obj from images as list of openCV object
#save target obj in ./img/sub/
target_obj = extract_obj.extract_obj_from_images(images)