import cv2
import os
import pandas as pd
import numpy as np
import uuid
from PIL import Image
import face_recognition as fr
import sqlite3
import pymongo
import base64
from flask import Flask, request,jsonify, make_response
from flask_cors import CORS

from support import MongoSupport

# MONGO_URL="mongodb://10.11.12.238:27017"
MONGO_URL="mongodb+srv://instaclone:instaclone@cluster0.iefhcu5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME="insta"

app = Flask(__name__)
CORS(app)
ms=MongoSupport()
success,db = ms.returnDb(MONGO_URL,db_name=DB_NAME)
if not success:exit(1)


# taking care of directories
face_ds_path = r"./face_imgs"
post_ds_path = r"./post_imgs"
post_prediction_path = r"./post_preds"
os.makedirs(face_ds_path, exist_ok=True)
os.makedirs(post_ds_path, exist_ok=True)
os.makedirs(post_prediction_path, exist_ok=True)

@app.get("/")
def homePage() :
    message = """
    <htmL>
    <h1> HomePage</h1>
    </html>
    """
    return message

@app.post("/register-face")
def registerFace(face_img_path="",user_id="unidentified"):
    data = request.form
    object_id = data.get("object_id")
    image_data = data.get("pic")
    user_id = data.get("user_id")
    print(object_id, user_id)
    
    print("[+] Face Registration In Progress. User Id :", user_id)
    response, status_code=saveImage(image_data, face_ds_path)
    if status_code: return response, status_code
    else: face_img_path = response
    try:
        print("face_img_path", face_img_path)
        # img = cv2.imread(face_img_path)

        face_img = fr.load_image_file(face_img_path)
        face_locations=detectFaces(face_img_path)
        print("[+] Training...")
        face_encoding = fr.face_encodings(face_img,face_locations,num_jitters=25,model="large")[0]
        # data = {"id":user_id, "face_encoding":face_encoding}

        # saving face encoding to db
        ms.addFaceEncoding(db, user_id, face_encoding.tobytes())
        print("[+] Data Saved Successfully...")
        os.remove(face_img_path)
        return {"message":"Saved"},200

    except Exception as e:
        print("[-] Error while registering face !!!",e)
        return {"message":"ERROR"},500


@app.post("/get-suggestions")
def returnSuggestions():
    data = request.form
    image_data = data.get("pic")
    # face_img = data.get("pic")
    user_id = data.get("user_id")    
    response, status_code=saveImage(image_data, post_ds_path)
    print(response, status_code)
    if status_code: return response, status_code
    else: post_img_path = response

    image = cv2.imread(post_img_path)

    # getting followings of the poster
    followings_object_ids=ms.getFollowings(db, user_id)
    user_ids, db_face_encodings = ms.getFaceEncodings(db,followings_object_ids).values()
    if not db_face_encodings:
        print("[!] No Face Data For The Poster's Following")
        return [],200
    face_locations=[]
    face_classifier=cv2.CascadeClassifier(r"./cv2_classifiers/haarcascade_frontalface_alt2.xml")
    faces = face_classifier.detectMultiScale(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5) # output (x, y, w, h)

    print(f"[+] {len(faces)} faces detected.")
    for (x,y,w,h) in faces:
        # print(x,y,w,h)
        #expanding the bounding box by 7 percent from each side. that is over all 14 percent.
        x_offset = int(w * 0.1)  
        y_offset = int(h * 0.1)   
        x = max(0, x - x_offset)  
        y = max(0, y - y_offset)  
        w = w + 2 * x_offset
        h = h + 2 * y_offset
        
        top = y
        right = x + w
        bottom = y + h
        left = x
        face_locations.append([top, right, bottom, left])
    
    image_face_encodings = fr.face_encodings(image, face_locations,num_jitters=5,model="large")
    # print("image_face_encodings:", image_face_encodings)
    im = image.copy()
    output=[]
    for (x, y, w, h), (top, right, bottom, left), cropped_face_encoding in zip(faces,face_locations, image_face_encodings):
        (x, y, w, h)=(int(x), int(y), int(w), int(h))
        tolerance=0.58
        # matches = fr.compare_faces(db_face_encodings, cropped_face_encoding,tolerance=tolerance)
        # if True in matches:
        # best_match_index = matches.index(True)
        face_distances = fr.face_distance(db_face_encodings, cropped_face_encoding)
        best_match_index = np.argmin(face_distances)
        print("best_match_index:", best_match_index)
        dic={}
        distance=face_distances[best_match_index]
        name = user_ids[best_match_index]
        print(name, distance)
        print(f"typeof distance: {type(distance)}")
        if distance<=tolerance:
            existing_entry = next((dic for dic in output if dic["name"] == name), None)
            if existing_entry:
                if existing_entry["distance"]>distance:
                    existing_entry["distance"]=float(distance)
                    existing_entry["bbox"]=(x,y,w,h)
            else:
                dic["name"]=name
                dic["distance"]=float(distance)
                dic["bbox"]=(x,y,w,h)
                print(dic)
                output.append(dic)
            
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
    for person in output:
        x,y,w,h = person["bbox"]
        name = person["name"]
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(im,name,(x, y+h), cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),2)
    
    save_path = r"./post_preds/"+"res.jpg"
    print(save_path)
    cv2.imwrite(save_path, im)
    print("saved to ", save_path)
    os.remove(post_img_path)

    print("output: ", output)
    response={"result":output}
    
    return jsonify(output), 200

def detectFaces(face_img_path)->list:
    image = cv2.imread(face_img_path, cv2.COLOR_BGR2GRAY)
    face_locations=[]
    face_classifier=cv2.CascadeClassifier(r"./cv2_classifiers/haarcascade_frontalface_alt2.xml")
    faces = face_classifier.detectMultiScale(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5) # output (x, y, w, h)

    print(f"[+] {len(faces)} faces detected.")
    for (x,y,w,h) in faces:
        # print(x,y,w,h)
        #expanding the bounding box by 7 percent from each side. that is over all 14 percent.
        x_offset = int(w * 0.1)  
        y_offset = int(h * 0.1)   
        x = max(0, x - x_offset)  
        y = max(0, y - y_offset)  
        w = w + 2 * x_offset
        h = h + 2 * y_offset
        
        top = y
        right = x + w
        bottom = y + h
        left = x
        face_locations.append([top, right, bottom, left])
    return face_locations

def saveImage(image_data,save_folder_path):
    print("[+] Saving Bytes To Image ...")
    # Converting encoded string to image
    try:
        print("imagedata:", image_data[:20])
        header, encoded_img_data = image_data.split(',', 1)
        format = header.split('/')[1].split(";")[0]
    except ValueError:
        message="Error: Invalid image data format"
        print(message)
        return message, 400

    # Check if the format is supported (modify if needed)
    if format not in ['jpeg', 'jpg', 'png']:
        message = "Error: Unsupported image format"
        print(message)
        return message, 400

    # Decode the base64 data
    try:
        decoded_img_data = base64.b64decode(encoded_img_data)
    except Exception as e:
        return f"Error: Decoding image data failed: {e}", 400

    # Generate a unique filename    
    post_img_path = os.path.join(post_ds_path, str(uuid.uuid4())+"."+format)
    with open(post_img_path, "wb") as f:
        f.write(decoded_img_data)
    
    return post_img_path, None
    


def getFaceEncodings():
    with sqlite3.connect("face_data.db") as db:
        cursor = db.cursor()
        query = """SELECT user_id, face_encoding FROM faces;"""
        cursor.execute(query)
        arr=cursor.fetchall()
        db.commit()
    db.close()
    data={}
    user_ids=[]
    face_encodings=[]
    for id,(user_id, face_encoding) in enumerate(arr):
        user_ids.append(user_id)
        face_encodings.append(np.frombuffer(face_encoding))
        # d={"id":id, "user_id":user_id, "face_encoding":np.frombuffer(face_encoding)}
        # data.append(d)
    data={"user_ids":user_ids, "face_encodings":face_encodings}
    # print(data)
    return data



def main():
    # registerFace(os.path.join(face_ds_path,"@sankalp.jpg"), "@sankalp")
    # registerFace(os.path.join(face_ds_path,"@nidhi.jpeg"), "@nidhi")
    # registerFace(os.path.join(face_ds_path,"@amitesh.jpg"), "@amitesh")
    # registerFace(os.path.join(face_ds_path,"@anish.jpg"), "@anish")
    # registerFace(os.path.join(face_ds_path,"@bhagyashree.jpg"), "@bhagyashree")
    # registerFace(os.path.join(face_ds_path,"@om.jpg"), "@om")
    # registerFace(os.path.join(face_ds_path,"@avikalp.jpg"), "@avikalp")
    # registerFace(os.path.join(face_ds_path,"@rishabh.jpg"), "@rishabh")
    
    # print(os.path.join(face_ds_path,"@rishabh.jpg"))
    # getFaceEncodings()

    # for i in range(1,7):
    #     path = fr"./group_photos/{i}.jpg"
    #     print("Current Image: ",path)
    #     returnSuggestions(path)
    
    # path = fr"./faces/@om.jpg"
    # returnSuggestions(path)
    pass




if __name__=="__main__":
    app.run(host="0.0.0.0",port=4488)#,debug=True)
    # main()