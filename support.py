import pymongo
import numpy as np

class MongoSupport:
    def returnDb(self, MONGO_URL,db_name):
        try:
            print("[+] Connecting To DB...")
            client = pymongo.MongoClient(MONGO_URL)
            db = client[db_name]
            print("[+] Connected.")
            return True, db
        except Exception as e:
            print("[-] Error connecting to monogo db",e)
            return False, None
    
    def addFaceEncoding(self, db, user_id, face_encoding):
        print("[+] Pushing Encoding To DB...")
        try:
            collection=db["users"]
            match_field={"name":user_id}
            set_query={"$set": {
                        "faceEncoding": face_encoding}}
            
            collection.update_one(match_field,set_query,upsert=False)
        except Exception as e:
            print("[-] Error ", e)


    def getFaceEncodings(self, db, object_id_list):
        print("[+] Fetching Face Details Of the Followings...")
        user_ids=[]
        face_encodings=[]
        collection=db["users"]
        try:
            for object_id in object_id_list:
                aggr_query=[
                {
                    "$match":
                    {"_id": object_id}
                },
                {
                    "$project":
                    {
                        "_id":0,
                        "name":1,
                        "faceEncoding":1}
                },
            ]
                row=collection.aggregate(aggr_query)
                if row:
                    data=list(row)
                    if len(data[0].keys())>1:
                        user_ids.append(data[0]["name"])
                        face_encodings.append(np.frombuffer(data[0]["faceEncoding"]))
            ret = {"user_ids":user_ids, "face_encodings":face_encodings}
            # print(ret)
            return ret

        except Exception as e:
            print("[-] Error ", e)
            return {}


    def getFollowings(self, db, user_id)->list:
        print("[+] Fetching Followings of the user", user_id)
        try:
            collection=db["users"]
            aggr_query=[
                {
                    "$match":
                    {"name": user_id}
                },
                {
                    "$project":
                    {
                        "_id":0,
                        "following": 1}
                },
            ]
            
            rows=collection.aggregate(aggr_query)
            data = list(rows)
            res=[]
            try:
                res = data[0]["following"]
                return res
            except Exception as e:
                print(e)
                return res
        except Exception as e:
            print("[-] Error",e)
            return res




if __name__=="__main__":
    # MONGO_URL="mongodb://10.11.12.238:27017"
    MONGO_URL="mongodb+srv://instaclone:instaclone@cluster0.iefhcu5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    DB_NAME="insta"

    ms = MongoSupport()
    success,db = ms.returnDb(MONGO_URL, DB_NAME)
    object_ids = ms.getFollowings(db, "nitin")
    print(object_ids)

    db_face_encodings=ms.getFaceEncodings(db,object_id_list=object_ids)
    print("db_face_encodings:", db_face_encodings)



