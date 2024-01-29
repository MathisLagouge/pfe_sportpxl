#import
import cv2 as cv
import os


#get img test with opencv
def get_img_test(directory):

    if os.path.exists(directory):
        # Get a list of all files in the directory
        file_list = os.listdir(directory)

        # Filter the list to only include jpg
        image_files = [f for f in file_list if f.lower().endswith(('.jpg'))]

        # Read and process each image in the directory
        images = []
        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            images.append(cv.imread(image_path))
    
    return images


#return a list of coord [[x1,y1,x2,y2],[x1,y1,x2,y2],...]
def get_coord_from_predict(predict):
    
    for p in predict:
        print(p)
        list = p.boxes.xyxy.squeeze().tolist()
        if (type(list[0])!=type([])):
            list = [list]
        coord = []
        for l in list:
            coord.append([int(x) for x in l])

    return coord


#extract sub_pic from pic
#return a list of sub_pic
#and resize (add a padding of 10%)
def extract_sub(img,coord, size):
    sub_img = []
    for c in coord:
        x1, y1 = c[0], c[1]
        x2, y2 = c[2], c[3]

        #add appropriete padding, with a security of 10%
        h = abs(y1-y2)
        w = abs(x1-x2)
        padding_1 = abs(h-w)//2
        if w < h:
            padding_2 = h//20
            x1 = x1 - padding_1 - padding_2
            x2 = x2 + padding_1 + padding_2
            y1 = y1 - padding_2
            y2 = y2 + padding_2
        else:
            padding_2 = w//20
            x1 = x1 - padding_2
            x2 = x2 + padding_2
            y1 = y1 - padding_1 - padding_2
            y2 = y2 + padding_1 + padding_2

        resized_img = cv.resize(img[y1:y2, x1:x2], size)
        sub_img.append(resized_img)

    return sub_img


#extract ans save object
#return a list of object as numpy array
def extract_obj_from_images(images, saving_img=False):

    #get model
    from yolo_model import yolo_model
    model = yolo_model()

    #save predict img
    #model.predict(images,save=True)

    #analyse each img
    target_obj = []
    j = 0
    for i in range(len(images)):

        img = images[i]

        #predict a img
        pred = model.predict(img)

        #get coord of find object
        coord = get_coord_from_predict(pred)

        #save sub img
        sub_img = extract_sub(img,coord, (512,512))
        for s in sub_img:
            target_obj.append(s)
            if saving_img:
                cv.imwrite('img/sub/sub'+ str(j) + '.jpg', s)
            j = j+1

    return target_obj