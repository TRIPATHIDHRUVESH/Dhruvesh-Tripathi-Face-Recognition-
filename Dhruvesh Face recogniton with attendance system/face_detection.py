
import cv2
import numpy as npy
import face_recognition as face_rec
# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# img declaration
Dhruvesh = face_rec.load_image_file('Sample_images\Dhruvesh.jpg')
Dhruvesh = cv2.cvtColor(Dhruvesh, cv2.COLOR_BGR2RGB)
Dhruvesh= resize(Dhruvesh, 0.50)
Dhruvesh_test = face_rec.load_image_file('Sample_images\Elon musk.jpg')
Dhruvesh_test = resize(Dhruvesh_test, 0.50)
Dhruvesh_test = cv2.cvtColor(Dhruvesh_test, cv2.COLOR_BGR2RGB)

# finding face location

faceLocation_Dhruvesh = face_rec.face_locations(Dhruvesh)[0]
encode_Dhruvesh = face_rec.face_encodings(Dhruvesh)[0]
cv2.rectangle(Dhruvesh, (faceLocation_Dhruvesh[3], faceLocation_Dhruvesh[0]), (faceLocation_Dhruvesh[1], faceLocation_Dhruvesh[2]), (255, 0, 255), 3)


faceLocation_Dhruveshtest = face_rec.face_locations(Dhruvesh_test)[0]
encode_Dhruveshtest = face_rec.face_encodings(Dhruvesh_test)[0]
cv2.rectangle(Dhruvesh_test, (faceLocation_Dhruvesh[3], faceLocation_Dhruvesh[0]), (faceLocation_Dhruvesh[1], faceLocation_Dhruvesh[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_Dhruvesh], encode_Dhruveshtest)
print(results)
cv2.putText(Dhruvesh_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', Dhruvesh)
cv2.imshow('test_img', Dhruvesh)
cv2.waitKey(0)
cv2.destroyAllWindows()