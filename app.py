import cv2
import os
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import face_recognition as fr
import sqlite3

face_ds_path = r"./faces"
# img = fr.load_image_file(os.path.join(face_ds_path,"@sankalp.jpg"))

def registerFace(face_img_path="",user_id="unidentified"):
    print("[+] Face Registration In Progress. User Id :", user_id)
    print("face_ds_path: ",face_ds_path)
    
    try:
        # first chek if the db exists or not and also if the use is present or not
        with sqlite3.connect("face_data.db") as db:
            cursor = db.cursor()
            schema = """ID INTEGER PRIMARY KEY AUTOINCREMENT, 
                    user_id NVARCHAR(30),
                    face_encoding BLOB """
            cursor.execute(f"CREATE TABLE IF NOT EXISTS faces ({schema})")
            db.commit()
            query = """SELECT 1 FROM faces WHERE user_id==(?);"""
            cursor.execute(query,(user_id,))
            already_present = cursor.fetchone()
        db.close()
        if already_present:
            print("[!] User face already registered.")
            return {"message":"Already Registered"}
        else:
            face_img = fr.load_image_file(face_img_path)
            # print(face_img)
            face_locations=detectFaces(face_img_path)
            print("[+] Training")
            face_encoding = fr.face_encodings(face_img,face_locations,num_jitters=70,model="large")[0]
            # data = {"id":user_id, "face_encoding":face_encoding}
            # print(data)
            #storing face_encoding and label in db
            with sqlite3.connect("face_data.db") as db:
                cursor = db.cursor()
                query = """ INSERT OR REPLACE INTO faces (user_id, face_encoding)
                            VALUES (?,?) 
                        """
                cursor.execute(query,(user_id, face_encoding.tobytes()))
                db.commit()
            db.close()
            print("[+] Data Saved Successfully...")
            return {"message":"Saved"}

    except Exception as e:
        print("[-] Error while registering face !!!",e)
        return {"message":"ERROR"},500

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




def returnSuggestions(image_path=None):
    if not image_path:
        return {"message":"No Image"},500
    else:
        # image = fr.load_image_file(image_path)
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    user_ids, db_face_encodings = getFaceEncodings().values()
    # print("db_face_encodings:",db_face_encodings)
    # print(type(db_face_encodings[0]))
    
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
    
    image_face_encodings = fr.face_encodings(image, face_locations,num_jitters=1,model="large")
    # print("image_face_encodings:", image_face_encodings)
    im = image.copy()
    output=[]
    for (x, y, w, h), (top, right, bottom, left), cropped_face_encoding in zip(faces,face_locations, image_face_encodings):
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
        if distance<=tolerance:
            existing_entry = next((dic for dic in output if dic["name"] == name), None)
            if existing_entry:
                if existing_entry["distance"]>distance:
                    existing_entry["distance"]=distance
                    existing_entry["bbox"]=(x,y,w,h)
            else:
                dic["name"]=name
                dic["distance"]=distance
                dic["bbox"]=(x,y,w,h)
                print(dic)
                output.append(dic)
            
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
    for person in output:
        x,y,w,h = person["bbox"]
        name = person["name"]
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(im,name,(x, y+h), cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),2)
    
    print(output)
    save_path = r"./predictions/"+image_path.split("/")[-1]
    cv2.imwrite(save_path, im)
    print("saved to ", save_path)
    return True



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
    main()