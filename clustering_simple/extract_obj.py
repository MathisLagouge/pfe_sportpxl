#import
import os
import cv2 as cv
from ultralytics import YOLO


#creat an image class
#img : numpy array
#name : key
class MyImage:
    def __init__(self, img_path, img_name):
        self.img = cv.imread(img_path)
        self.__name = os.path.splitext(img_name)[0]

    def __str__(self):
        return self.__name
    
    def get_img(self):
        return self.img
    


#get img test with opencv
def get_img_from_dir(directory):

    if os.path.exists(directory):
        # Get a list of all files in the directory
        file_list = os.listdir(directory)

        # Filter the list to only include jpg
        image_files = [f for f in file_list if f.lower().endswith(('.jpg')) or f.lower().endswith(('.png'))]

        # Read and process each image in the directory
        images = []
        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            images.append(MyImage(image_path, image_file))
            print("find " + str(image_file) + " ! ")
        
        return images

    else:
        return "path not found"

    
    


#return a list of coord [[x1,y1,x2,y2],[x1,y1,x2,y2],...]
def get_coord_from_predict(predict):
    
    coord = []
    for p in predict:

        list = p.boxes.xyxy.squeeze().tolist()
        if list!=[]:
            if (type(list[0])!=type([])):
                list = [list]
            cls = p.boxes.cls.squeeze().tolist()
            if (type(cls)!=type([])):
                cls = [cls]

            for i in range(len(list)):
                #cars id with yolov8 is 2.0
                if cls[i] == 4:
                    coord.append([int(x) for x in list[i]])

    return coord


#extract sub_pic from pic
#return a list of sub_pic
#and resize (add a padding of 10%)
def extract_sub(img,coord, size):
    sub_img = []
    if coord!=[]:
        for c in coord:
            x1, y1 = c[0], c[1]
            x2, y2 = c[2], c[3]

            resized_img = cv.resize(img[y1:y2, x1:x2], size)
            sub_img.append(resized_img)

        if sub_img != []:
            a_max = sub_img[0].shape[0] * sub_img[0].shape[1]
            max = sub_img[0]
            for s in sub_img:
                a = s.shape[0] * s.shape[1]
                if a > a_max:
                    a_max = a
                    max = s
            return max
        else:
            return []
    else:
        return []


#extract and save object
#image are classe from sort_img.py
#return a list of object as numpy array
def extract_obj_from_images(images, size, model_path, saving_img=False, saving_img_path="./"):

    #get model
    model = YOLO(model_path)

    #analyse each img
    target_obj = []
    for i in range(len(images)):
        
        #get img as numpy array
        img = images[i]

        #predict img
        pred = model.predict(MyImage.get_img(img))

        #get coord of find object
        coord = get_coord_from_predict(pred)

        #extract tha car find on the picture
        sub_img = extract_sub(MyImage.get_img(img),coord, size)

        #remove background


        #save the car
        if sub_img != []:
            target_obj.append(sub_img)
            if saving_img:
                img_path = saving_img_path + str(img) + '.jpg'
                cv.imwrite(img_path, sub_img)
                print(img_path + '  saved !')

    return target_obj