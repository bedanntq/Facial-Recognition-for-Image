# # USAGE
# # python src/recoginzie_face/recognize_faces_image.py --encodings encodings.pickle --image test_images/anhtest.png
# import face_recognition
# import argparse
# import pickle
# import cv2

# ap = argparse.ArgumentParser()
# # đường dẫn đến file encodings đã lưu
# ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
# ap.add_argument("-i", "--image", required=True, help="path to the test image")
# # nếu chạy trên CPU hay embedding devices thì để hog, còn khi tạo encoding vẫn dùng cnn cho chính xác
# ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face dettection model to use: cnn or hog")
# args = vars(ap.parse_args())

# # load the known faces and encodings
# print("[INFO] loading encodings...")
# data = pickle.loads(open(args["encodings"], "rb").read())      # loads - load từ file

# # load image và chuyển từ BGR to RGB (dlib cần để chuyển về encoding)
# image = cv2.imread(args["image"])
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # CŨng làm tương tự cho ảnh test, detect face, extract face ROI, chuyển về encoding
# # rồi cuối cùng là so sánh kNN để recognize
# print("[INFO] recognizing faces...")
# boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
# encodings = face_recognition.face_encodings(rgb, boxes)

# # khởi tạo list chứa tên các khuôn mặt phát hiện được
# # nên nhớ trong 1 ảnh có thể phát hiện được nhiều khuôn mặt nhé
# names = []

# # duyệt qua các encodings của faces phát hiện được trong ảnh
# for encoding in encodings:
#     # khớp encoding của từng face phát hiện được với known encodings (từ datatset)
#     # so sánh list of known encodings và encoding cần check, sẽ trả về list of True/False xem từng known encoding có khớp với encoding check không
#     # có bao nhiêu known encodings thì trả về list có bấy nhiêu phần tử
#     # trong hàm compare_faces sẽ tính Euclidean distance và so sánh với tolerance=0.6 (mặc định), nhó hơn thì khớp, ngược lại thì ko khớp (khác người)
#     matches = face_recognition.compare_faces(data["encodings"], encoding, 0.4)      # có thể điều chỉnh tham số cuối
#     name = "Unknown"    # tạm thời vậy, sau này khớp thì đổi tên

#     # Kiểm tra xem từng encoding có khớp với known encodings nào không,
#     if True in matches:
#         # lưu các chỉ số mà encoding khớp với known encodings (nghĩa là b == True)
#         matchedIdxs = [i for (i, b) in enumerate(matches) if b]

#         # tạo dictionary để đếm tổng số lần mỗi face khớp
#         counts = {}
#         # duyệt qua các chỉ số được khớp và đếm số lượng 
#         for i in matchedIdxs:
#             name = data["names"][i]     # tên tương ứng known encoding khiowps với encoding check
#             counts[name] = counts.get(name, 0) + 1  # nếu chưa có trong dict thì + 1, có rồi thì lấy số cũ + 1

#         # lấy tên có nhiều counts nhất (tên có encoding khớp nhiều nhất với encoding cần check)
#         # có nhiều cách để có thể sắp xếp list theo value ví dụ new_dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
#         # nó sẽ trả về list of tuple, mình chỉ cần lấy name = new_dic[0][0]
#         name = max(counts, key=counts.get)

#     names.append(name)

# # Duyệt qua các bounding boxes và vẽ nó trên ảnh kèm thông tin
# # Nên nhớ recognition_face trả bounding boxes ở dạng (top, rights, bottom, left)
# for ((top, right, bottom, left), name) in zip(boxes, names):
#     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
#     y = top - 15 if top - 15 > 15 else top + 15

#     cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

# cv2.imshow("Image", image)
# cv2.waitKey(0)

# ==============================================================================================================================================
# USAGE
# python src/recoginzie_face/recognize_faces_image.py --image test_images/anhtest.png



# import face_recognition
# import cv2
# import pickle
# import argparse
# import firebase_admin
# from firebase_admin import credentials, db
# from tabulate import tabulate

# # Argument parser for command-line arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the test image")
# ap.add_argument("-k", "--key", help="Optional: Key to query in Firebase (defaults to recognized name)")
# args = vars(ap.parse_args())

# # Default paths (change as per your project structure)
# ENCODINGS_PATH = "encodings.pickle"
# CREDENTIALS_PATH = "json/serviesAcc.json"
# FIREBASE_URL = "https://aidatalist-default-rtdb.asia-southeast1.firebasedatabase.app/"

# # Initialize Firebase
# print("[INFO] Initializing Firebase...")
# cred = credentials.Certificate(CREDENTIALS_PATH)
# firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
# ref = db.reference("/users")  # Reference to the 'users' node

# # Load facial encodings
# print("[INFO] Loading face encodings...")
# data = pickle.loads(open(ENCODINGS_PATH, "rb").read())

# # Load the test image
# print("[INFO] Recognizing faces in the image...")
# image = cv2.imread(args["image"])
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Detect faces and encode
# boxes = face_recognition.face_locations(rgb, model="cnn")  # Default to "hog"
# encodings = face_recognition.face_encodings(rgb, boxes)
# names = []

# for encoding in encodings:
#     matches = face_recognition.compare_faces(data["encodings"], encoding, 0.4)
#     name = "Unknown"
#     if True in matches:
#         matchedIdxs = [i for (i, b) in enumerate(matches) if b]
#         counts = {}
#         for i in matchedIdxs:
#             name = data["names"][i]
#             counts[name] = counts.get(name, 0) + 1
#         name = max(counts, key=counts.get)
#     names.append(name)

# # Draw boxes and names on the image
# for ((top, right, bottom, left), name) in zip(boxes, names):
#     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
#     y = top - 15 if top - 15 > 15 else top + 15
#     cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

# cv2.imshow("Image", image)
# cv2.waitKey(0)

# # Query Firebase for recognized names
# query_key = args["key"] or (names[0] if names else None)
# if query_key:
#     print(f"[INFO] Querying Firebase for '{query_key}'...")
#     data = ref.child(query_key).get()
#     if data:
#         if isinstance(data, dict):
#             table_data = [(key, value) for key, value in data.items()]
#         else:
#             table_data = [(query_key, data)]
#         headers = ["Field", "Value"]
#         print(tabulate(table_data, headers=headers, tablefmt="grid"))
#     else:
#         print(f"[INFO] No information found for '{query_key}' in Firebase.")
# else:
#     print("[INFO] No valid key provided for Firebase query.")



# ======================================================================================
# USAGE
# python src/recoginzie_face/recognize_faces_image.py --image test_images/anhtest.png

import face_recognition
import cv2
import pickle
import argparse
import firebase_admin
from firebase_admin import credentials, db
from tabulate import tabulate

# Argument parser for command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the test image")
ap.add_argument("-k", "--key", help="Optional: Key to query in Firebase (defaults to recognized name)")
ap.add_argument("-o", "--output", default="output.txt", help="Path to the output text file")
args = vars(ap.parse_args())

# Default paths (change as per your project structure)
ENCODINGS_PATH = "encodings.pickle"
CREDENTIALS_PATH = "json/serviesAcc.json"
FIREBASE_URL = "https://aidatalist-default-rtdb.asia-southeast1.firebasedatabase.app/"

# Initialize Firebase
print("[INFO] Initializing Firebase...")
cred = credentials.Certificate(CREDENTIALS_PATH)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
ref = db.reference("/users")  # Reference to the 'users' node

# Load facial encodings
print("[INFO] Loading face encodings...")
data = pickle.loads(open(ENCODINGS_PATH, "rb").read())

# Load the test image
print("[INFO] Recognizing faces in the image...")
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces and encode
boxes = face_recognition.face_locations(rgb, model="cnn")  # Default to "hog"
encodings = face_recognition.face_encodings(rgb, boxes)
names = []

for encoding in encodings:
    matches = face_recognition.compare_faces(data["encodings"], encoding, 0.4)
    name = "Unknown"
    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        name = max(counts, key=counts.get)
    names.append(name)

# Draw boxes and names on the image
for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

# Save the image with bounding boxes (optional)
output_image_path = "output_image.jpg"
cv2.imwrite(output_image_path, image)

# Query Firebase for recognized names
query_key = args["key"] or (names[0] if names else None)
output_lines = []  # To collect output lines for writing to file
if query_key:
    print(f"[INFO] Querying Firebase for '{query_key}'...")
    data = ref.child(query_key).get()
    if data:
        output_lines.append(f"Results for '{query_key}':")
        if isinstance(data, dict):
            table_data = [(key, value) for key, value in data.items()]
        else:
            table_data = [(query_key, data)]
        headers = ["Field", "Value"]
        output_lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(output_lines[-1])
    else:
        message = f"[INFO] No information found for '{query_key}' in Firebase."
        print(message)
        output_lines.append(message)
else:
    message = "[INFO] No valid key provided for Firebase query."
    print(message)
    output_lines.append(message)

# Write to text file immediately
with open(args["output"], "w") as f:
    for line in output_lines:
        f.write(line + "\n")
    print(f"[INFO] Results written to {args['output']}")

cv2.imshow("Image", image)
cv2.waitKey(0)

