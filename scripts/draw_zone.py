import cv2
import numpy as np

# لیست برای ذخیره نقاط کلیک شده
points = []

def mouse_callback(event, x, y, flags, param):
    """تابع مدیریت رویدادهای موس"""
    global points
    
    # اگر کلیک چپ شد، نقطه را ذخیره کن و دایره بکش
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point selected: [{x}, {y}]")

def main():
    video_path = "data/input/test_video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video at {video_path}")
        return

    # خواندن فریم اول برای رسم
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
        
    clone = frame.copy()
    window_name = "Zone Drawing Tool (Click to draw, 'c' to clear, 's' to save/finish)"
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n--- INSTRUCTIONS ---")
    print("1. Click on the image to draw the polygon corners.")
    print("2. Press 'c' to clear all points.")
    print("3. Press 's' to save the coordinates and exit.")
    print("--------------------\n")

    while True:
        # کپی فریم برای پاک شدن و رسم مجدد
        display = clone.copy()
        
        # رسم نقاط و خطوط بین آن‌ها
        if len(points) > 0:
            for pt in points:
                cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
                
            if len(points) > 1:
                pts_array = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts_array], isClosed=False, color=(255, 0, 0), thickness=2)
                
            # رسم خط پایانی اگر حداقل ۳ نقطه داشته باشیم (برای نمایش حالت بسته)
            if len(points) >= 3:
                cv2.line(display, tuple(points[-1]), tuple(points[0]), (255, 0, 0), 2)
                
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            points.clear()
            print("Points cleared.")
        elif key == ord('s'):
            break

    cv2.destroyAllWindows()
    cap.release()
    
    # چاپ خروجی به فرمت YAML برای کپی راحت
    if len(points) >= 3:
        print("\n--- COPY THIS INTO YOUR settings.yaml ---")
        print("zones:")
        print("  - id: \"machine_01\"")
        print("    name: \"Machine 01\"")
        print("    polygon:")
        for pt in points:
            print(f"      - [{pt[0]}, {pt[1]}]")
        print("-----------------------------------------")
    else:
        print("\nYou need at least 3 points to form a zone polygon. Nothing saved.")

if __name__ == "__main__":
    main()