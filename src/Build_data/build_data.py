# Usage:
# python src/build_data/build_data.py --o data/quy

""" Xây dựng dataset bằng cách chụp webcam """
import cv2
import os
import argparse

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có khởi tạo được không nếu không khởi tạo được thì thông báo lỗi
if not cap.isOpened():
    print("Lỗi: Không thể mở camera.")
    exit()

# Tạo parser để xử lý tham số dòng lệnh
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(ap.parse_args())

# Kiểm tra và tạo thư mục nếu chưa tồn tại
output_dir = args["output"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f'[INFO] Thư mục {output_dir} đã được tạo ra')

# Mở camera và bắt đầu xử lý
video = cv2.VideoCapture(0)
total = 0

while True:
    ret, frame = video.read()
    
    cv2.imshow("video", frame)
    key = cv2.waitKey(1) & 0xFF

    # Nhấn 'k' để lưu ảnh
    if key == ord("k"):
        # Đảm bảo thư mục tồn tại trước khi lưu từng ảnh
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"[INFO] Output directory {output_dir} re-created.")
            
        # Ghi ảnh vào thư mục
        p = os.path.sep.join([output_dir, "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, frame)
        total += 1
    
    # Nhấn 'q' để thoát
    elif key == ord("q"):
        break

print("[INFO] {} face images stored".format(total))
video.release()
cv2.destroyAllWindows()


