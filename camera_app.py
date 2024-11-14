import cv2
import numpy as np
import torch
import pandas as pd
import face_recognition
import matplotlib.pyplot as plt
import os
import time
import csv
import argparse

os.environ["GST_DEBUG"] = "0"


class Face:
    """Class for worked camera on jetson nano"""
    def __init__(self, sensor_id=0, cap_width = 780, cap_height=480, d_width=780, d_height=480, framerate=30, flip_method=0):
        """Initialize setting for camera in class"""
        self.sensor_id=sensor_id
        self.capture_width=cap_width  #1920
        self.capture_height=cap_height  #1080
        self.display_width=d_width
        self.display_height=d_height
        self.framerate=framerate
        self.flip_method=flip_method
        self.name_csv = "emb.csv"
        self.path_root = "person/"
        #create csv if not file
        if not os.path.isfile(os.path.join(self.path_root, self.name_csv)):
            with open(os.path.join(self.path_root, self.name_csv), 'w', newline='') as file:
                writer = csv.writer(file)
                field = ['name', 'embedding']
                writer.writerow(field)

    def gstreamer_pipeline(self):
        """Initialize from gstreamer driver camera"""
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                self.sensor_id,
                self.capture_width,
                self.capture_height,
                self.framerate,
                self.flip_method,
                self.display_width,
                self.display_height,
            )
        )
    
    def save_face(self, name=""):
        """write embedding face and name one a person for recognition to csv file"""
        embed_person = 0
        cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER) # opened camera
        print(self.gstreamer_pipeline())
        print(f'CAP: ', cap.isOpened())
        try:
            if cap.isOpened():
                ret, frame = cap.read() # get image from camera
                if ret:
                    frame_rz = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    img = cv2.cvtColor(frame_rz, cv2.COLOR_BGR2RGB)
                    # process name
                    if name == "":
                        name = 'Not input Name'
                    else:
                        name = name
                    # process image
                    location_face = face_recognition.face_locations(img)
                    embedding = face_recognition.face_encodings(img, location_face)
                    load_faces = pd.read_csv(os.path.join(self.path_root, self.name_csv), converters={'embedding': eval}).reset_index()
                    names = list(load_faces['name'])
                    if embedding:
                        embed_person = embedding[0]
                        if load_faces.size > 0:
                            for i in range(load_faces.shape[0]):
                                save_emb_faces = np.array(load_faces['embedding'].iloc[i], dtype=np.float64)
                            
                                matches = face_recognition.compare_faces(embed_person[None,], save_emb_faces[None,]) #!
                                if matches and name in names:
                                    print(f'Such a person with name {name} is face already exists!!!')
                                    break
                        else:
                        
                            print(f'Ok embedding by a name {name}!')
                            if name != "" and embed_person.size > 0:
                    
                                data = pd.DataFrame({
                                    "name": [name],
                                    "embedding": [embed_person.tolist()]
                                })
                                data.to_csv(os.path.join(self.path_root, self.name_csv), index=False, mode='a',
                                            header=not os.path.exists(os.path.join(self.path_root, self.name_csv)))
                            plt.imshow(frame[:,:,::-1])
                            plt.title(f'{name}')
                            plt.show()
                else:
                        embed_person = 0
                        print(f'Not getting embebbing face for name: {name}(')
                    
        except Exception as e:
            print("Not opened camera, error: ", e)
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
        # return self.save_embeddings
    def show_camera(self):
        """Inference imaging recognizing face"""
        video = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        print(video.isOpened())
        if video.isOpened():
            try:
                while True:
                    ret, frame = video.read()
                    if not ret:
                        print("Not get frame!")
                        break

                    # frame resize
                    frame_rsz = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

                    # frame convert from bgr to rgb
                    frame_rgb = cv2.cvtColor(frame_rsz, cv2.COLOR_BGR2RGB)
 
                    # face_recognition all person  
                                       
                    locations_face = face_recognition.face_locations(frame_rgb)
                    # if locations_face:
                    # print(locations_face)
                    embeddings_face = face_recognition.face_encodings(frame_rsz, locations_face)

                    # read saves  embeddings faces
                    faces_name = []
                    get_location_faces = [] #!!!!!!!!!!
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    save_embed = 0
                    loaded_data = pd.read_csv(os.path.join(self.path_root, self.name_csv), converters={'embedding': eval})
                    for embedding_face, location_face in zip(embeddings_face, locations_face):
                        found_match = False # flag for matching face
                        for i in range(loaded_data.shape[0]):
                            name = loaded_data['name'].iloc[i]
                            save_embed = np.array(loaded_data['embedding'].iloc[i], dtype = np.float64)

                            #matching embeddings
                            matches_face = face_recognition.compare_faces(embedding_face[None], save_embed[None,], tolerance=0.6)
                            if matches_face[0]:
                                faces_name.append(name)
                                get_location_faces.append(location_face)
                                found_match = True
                                break
                        # if not matching found, face as Unknow
                        if not found_match:
                            faces_name.append('Unknow')
                            get_location_faces.append(location_face)

                    for (top, right, bottom, left), name_v in zip(get_location_faces, faces_name):

                        top *= 5
                        right *= 5
                        bottom *= 5
                        left *= 5
                        color = (0, 255, 0) if name_v != "Unknow" else (255, 0, 0)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
                        cv2.putText(frame, name_v, (left, top - 10), font, 1.0, (255, 255, 255), 1)
                
                    cv2.imshow(f'Video', frame)
                   
                    keyCode = cv2.waitKey(10) & 0xFF
                    if keyCode == 27 or keyCode == ord('q'):
                        break

            # except Exception as e:
            #     print("Not open camera gstreamer for video exception: ", e)
            finally:
                video.release()
                cv2.destroyAllWindows()

    def delete_face(self, name=""):
        path_csv = os.path.join(self.path_root, self.name_csv)
        df_faces = pd.read_csv(path_csv, index_col=0).reset_index()
        len_start = df_faces.shape[0]
        # get index
        if name != "":
            idx = df_faces[df_faces['name'] == f'{name}'].index
            df_faces.drop(index=idx, inplace=True)
            len_end = df_faces.shape[0]
            if len_end < len_start:
                print(f"Delet {name} OK!")
            else:
                print(f'Not {name} deleted, may be there is no such name in save!')
            df_faces.to_csv(path_csv, index = False)
        else:
            print("Write name in delete_face('name') !")



def main():
    parser = argparse.ArgumentParser(description="Programm for work with camera.")
    parser.add_argument("--video", action="store_true", help="Starts the video from the camera!") # for out video enter key 'q' or 'esc'
    parser.add_argument("--photo", type=str, help="Create photo and input name  for  save image witn a name")
    parser.add_argument("--delete_face", help="input name for deleted photo from base by with input name")

    args = parser.parse_args()

    face = Face()

    if args.video:
        face.show_camera()
    if args.photo:
        face.save_face(args.photo)
    if args.delete_face:
        face.delete_face(args.delete_face)

if __name__ == '__main__':
    main()