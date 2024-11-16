Demo kéo búa bao bằng mediapipe
1.Cài đặt thư viện Mediapipe và OpenCV
Tạo file setup chứa các thư viện trên
mediapipe
opencv-python
Cài đặt thư viện có trong file setup

pip install -r setup

Tìm kiếm hình ảnh dùng để training AI
Ví dụ:

![image](https://github.com/user-attachments/assets/ccb786ad-cdac-4116-aac9-1d4bc262d786)

![image](https://github.com/user-attachments/assets/e3a66ef2-dee7-4db7-91ff-03d6db072bfa)

![image](https://github.com/user-attachments/assets/6675090f-1ece-45d1-b229-3dea9e121436)


Đặt tên cho từng file lần lượt là 0 1 và 2 để hiển thị lên trên giao diện của người dùng
Xác định giá trị của user_draw dựa trên số n_fingers đếm được để biểu thị lựa chọn bao,búa,kéo và hiển thị kết quả của trò chơi.
Nhấn q thì sẽ kết thúc vòng lặp,nhấn phím cách sẽ hiển thị kết quả của trò chơi.
4.Kết quả demo
Khi không có bàn tay trong hình
![image](https://github.com/user-attachments/assets/13e13ec6-141f-4ad2-8cd0-f6f0ed93ad93)

 
Khi đưa tay trái hoặc đưa sai cử chỉ

 
Hiển thị kết quả trò chơi 
![image](https://github.com/user-attachments/assets/9805f771-455c-423a-a80d-fb6cc825052b)

