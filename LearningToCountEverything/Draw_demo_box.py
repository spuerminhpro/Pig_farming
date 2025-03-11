import cv2
import os

drawing = False
start_x, start_y = -1, -1
boxes = []
img = None
img_display = None
img_with_boxes = None  

def update_img_with_boxes():
    """Vẽ boxes lên ảnh"""
    global img, img_with_boxes, boxes
    img_with_boxes = img.copy()
    for box in boxes:
        cv2.rectangle(img_with_boxes, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

def draw_bounding_box(event, x, y, flags, param):
    """Vẽ bounding box và crosshair lines."""
    global drawing, start_x, start_y, boxes, img, img_display, img_with_boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        # Cập nhật ảnh với boxes
        if img_with_boxes is None:
            update_img_with_boxes()
        img_display = img_with_boxes.copy()

        if drawing:
            # Vẽ bounding box
            cv2.rectangle(img_display, (start_x, start_y), (x, y), (0, 255, 0), 2)

        # Vẽ crosshair lines
        height, width, _ = img.shape
        cv2.line(img_display, (0, y), (width, y), (0, 0, 255), 1)
        cv2.line(img_display, (x, 0), (x, height), (0, 0, 255), 1)
        cv2.imshow("Draw Bounding Boxes", img_display)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        x1, x2 = min(start_x, end_x), max(start_x, end_x)
        y1, y2 = min(start_y, end_y), max(start_y, end_y)
        boxes.append([x1, y1, x2, y2])
        update_img_with_boxes()

        # Vẽ bounding box và crosshair lines
        img_display = img_with_boxes.copy()
        height, width, _ = img.shape
        cv2.line(img_display, (0, y), (width, y), (0, 0, 255), 1)
        cv2.line(img_display, (x, 0), (x, height), (0, 0, 255), 1)
        cv2.imshow("Draw Bounding Boxes", img_display)

def main(image_path):
    """Main function to load the image, handle drawing, and save boxes."""
    global img, img_display, img_with_boxes

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Khởi tạo ảnh với boxes và ảnh hiển thị
    img_with_boxes = img.copy()
    img_display = img.copy()

    # Lưu ảnh
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(os.path.dirname(image_path), f"{image_name}.txt")

    cv2.namedWindow("Draw Bounding Boxes")
    cv2.setMouseCallback("Draw Bounding Boxes", draw_bounding_box)

    print("Instructions:")
    print("- Left-click and drag to draw bounding boxes (boxes are added automatically).")
    print("- Press 's' to save all boxes to a text file.")
    print("- Press 'r' to reset all boxes.")
    print("- Press 'd' to delete the last box.")
    print("- Press 'q' to quit.")

    while True:
        cv2.imshow("Draw Bounding Boxes", img_display)
        key = cv2.waitKey(2) & 0xFF

        if key == ord('s'):
            if boxes:
                with open(output_path, 'w') as f:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        f.write(f"{x1} {y1} {x2} {y2}\n")
                print(f"Saved {len(boxes)} bounding boxes to {output_path}")
            else:
                print("No boxes to save.")

        elif key == ord('r'):
            boxes.clear()
            update_img_with_boxes()
            img_display = img_with_boxes.copy()
            print("Reset all bounding boxes.")

        elif key == ord('d'):
            if boxes:
                boxes.pop()
                update_img_with_boxes()
                img_display = img_with_boxes.copy()
                print(f"Removed the last box. {len(boxes)} boxes remaining.")
            else:
                print("No boxes to remove.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r'C:\Users\phann\Documents\Pig_farming\LearningToCountEverything\output_frame\45\images\198.jpg'
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
    else:
        main(image_path)
