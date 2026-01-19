import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Hệ thống Phát hiện Gian lận", layout="wide")
st.title("🚫 AI Detection: Hệ thống Phát hiện Gian lận")
st.sidebar.header("Cấu hình mô hình")

# --- TẢI MÔ HÌNH ---
@st.cache_resource
def load_model():
    return YOLO('/Users/softann/Documents/BTDL_A47331_A47548_A49275_A47952/best.pt')  # Đảm bảo file best.pt nằm cùng thư mục

model = load_model()

# --- THANH CÔNG CỤ BÊN TRÁI ---
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.0, 1.0, 0.5)
blocked_ids = st.sidebar.multiselect(
    "Chặn các Class ID (Không hiển thị):", 
    options=list(model.names.keys()), 
    format_func=lambda x: model.names[x]
)

source_type = st.sidebar.radio("Chọn loại tệp tin:", ("Ảnh", "Video"))
uploaded_file = st.sidebar.file_uploader(f"Tải lên {source_type.lower()}", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])

# --- XỬ LÝ ẢNH ---
if source_type == "Ảnh" and uploaded_file:
    image = Image.open(uploaded_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    img_array = np.array(image)
    
    # Predict
    results = model.predict(img_array, conf=conf_threshold)
    
    # Vẽ kết quả (Sử dụng hàm plot của YOLO cho nhanh hoặc tùy biến như code cũ của bạn)
    res_plotted = results[0].plot() # Bạn có thể viết lại hàm vẽ riêng nếu muốn lọc blocked_ids
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Kết quả đánh giá")
        st.image(res_plotted, channels="BGR", use_container_width=True)

# --- XỬ LÝ VIDEO ---
elif source_type == "Video" and uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st.subheader("Luồng xử lý Video")
    frame_window = st.image([]) # Tạo một khung trống để cập nhật video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Xử lý YOLO
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        
        # Vẽ thủ công để áp dụng blocked_ids (giống code của bạn)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in blocked_ids:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # Màu đỏ cho gian lận
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Chuyển màu từ BGR sang RGB để hiển thị trên Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

    cap.release()