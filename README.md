Demo kéo búa bao bằng mediapipe
1.Cài đặt thư viện Mediapipe và OpenCV
Tạo file setup chứa các thư viện trên
mediapipe
opencv-python
Cài đặt thư viện có trong file setup

pip install -r setup

Tìm kiếm hình ảnh dùng để training AI
![image](https://github.com/user-attachments/assets/ccb786ad-cdac-4116-aac9-1d4bc262d786)
Bao	
![image](https://github.com/user-attachments/assets/e3a66ef2-dee7-4db7-91ff-03d6db072bfa)
Búa 
![image](https://github.com/user-attachments/assets/6675090f-1ece-45d1-b229-3dea9e121436)
Kéo

Đặt tên cho từng file lần lượt là 0 1 và 2 để hiển thị lên trên giao diện của người dùng
2.Tạo thư viện để sử dụng thư viện mediapipe dùng để nhận diện bàn tay và đếm số ngón tay…
 
 
Khai báo thư viện OpenCV và mediapipe
 .
Tạo class handDetector theo kiểu hướng đối tượng để dễ quản lí code 

  
Tạo module hands để phát hiện và theo dõi bàn tay trong Thư viện mediapipe.

 
Tạo đối tượng hands từ module của hand để có thể sử dụng để xử lý hình ảnh hoặc khung hình trong camera để phát hiện bàn tay và landmark chi tiết trên bàn tay.

 
Vẽ các đốt tay trên lòng bàn tay bằng thư viên mediapipe

 
Tạo hàm tìm tay

 
Chuyển từ BGR sang RGB do nhận từ camera là dạng RGB

 
Đưa kết quả vào thư viện Mediapipe và truyền vào list xem có bao nhiêu ngón tay

 
Kiểm tra xem có tay hay không nếu có thì vẽ landmark cho bàn tay


 
Trích ra tọa độ của các khớp của ngón tay trong đó :
 
Dùng để lấy bàn tay phát hiện đầu tiên trong hình 
 
Lấy chiều cao và chiều rộng trong ảnh 
 
Lặp qua từng landdmark trên bàn tay
 
Chuyển đổi tọa độ chuẩn hóa sang tọa độ thực tế bằng cách nhân với chiều cao và chiều rộng của ảnh
 
Tạo hàm count_finger truyền list hand_lms vào để đếm số ngón tay
 
Thứ tự của từng đốt ngón tay bắt đầu từ ngón cái (4),Ngón trỏ (8),Ngón giữa (12),Ngón danh (16),Ngón út (20).
n_finger là biến đếm số lượng ngón tay giơ lên.
 
Kiểm tra ngón cái xem có được giơ lên hay không
 
Kiểm tra 4 ngón còn lại có được giơ lên hay không

 
Toàn bộ đoạn này dùng để kiểm tra xem bàn tay có được giơ lên hay không nếu không thì trả về không có bàn tay

3,Tạo file game kéo búa bao và import các thư viên cần thiết 
 
Thư viện random chứa các hàm tạo số ngẫu nhiên.
Thư viện os cho phép tương tác với hệ điều hành.

 
Khởi tạo đối tượng detector sử dụng class handDetector trong thư viện handlib 
 
Mở camera
 
Đặt giá trị chiều cao và chiều rộng cho khung hình của camera
 
Mỗi khung hình sẽ được lật và điều chỉnh kích thước 
 
Truyền hình ảnh vào detector
 
Tạo hàm draw_results 
 
Random số từ 0 đến 2 ứng với bao,búa,kéo
 
Sử dụng thư viên OpenCV để hiển thị hình ảnh và văn bản  lên khung hình theo user.
 
Sử dụng thư viện OpenCV để hiển thị hình ảnh và văn bản theo computer

 
Kiểm tra và hiển thị kết quả là thắng,thua hay hòa.



 
Xác định giá trị của user_draw dựa trên số n_fingers đếm được để biểu thị lựa chọn bao,búa,kéo và hiển thị kết quả của trò chơi.

 
Nhấn q thì sẽ kết thúc vòng lặp,nhấn phím cách sẽ hiển thị kết quả của trò chơi.
4.Kết quả demo
 
Khi không có bàn tay trong hình

 
Khi đưa tay trái hoặc đưa sai cử chỉ

 
Hiển thị kết quả trò chơi 
![image](https://github.com/user-attachments/assets/9805f771-455c-423a-a80d-fb6cc825052b)

