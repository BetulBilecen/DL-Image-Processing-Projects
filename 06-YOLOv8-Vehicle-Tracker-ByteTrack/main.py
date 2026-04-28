"""
PROJECT: Real-Time Object Tracking with YOLOv8
MODEL: YOLOv8n (Nano)
TRACKER: ByteTrack
"""

# ----------------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ----------------------------------------------------------------------------
from ultralytics import YOLO  # YOLOv8 model library
import cv2                    # image and video processing

# ----------------------------------------------------------------------------
# 2. VARIABLE DEFINITIONS FOR ERROR HANDLING
# ----------------------------------------------------------------------------
cap = None  # video source, defined here for safe release in finally block
out = None  # video writer, defined here for safe release in finally block

try:
    # ----------------------------------------------------------------------------
    # 3. MODEL AND VIDEO SOURCE INITIALIZATION
    # ----------------------------------------------------------------------------
    model = YOLO("yolov8n.pt")        # load pre-trained YOLOv8 nano model
    video_path = r"IMG_5268.MOV"      # path to the input video file

    cap = cv2.VideoCapture(video_path)  # open video source

    if not cap.isOpened():
        raise FileNotFoundError(f"Video file could not be opened: {video_path}")

    # ----------------------------------------------------------------------------
    # 4. OUTPUT VIDEO CONFIGURATION
    # ----------------------------------------------------------------------------
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # original video width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # original video height
    fps    = cap.get(cv2.CAP_PROP_FPS)                # frames per second

    if fps <= 0:       # assign default value if FPS cannot be read
        fps = 30
        print("Warning: FPS could not be read, using default value of 30.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')                          # video codec
    out    = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))  # output video writer

    print("Tracking started... Press 'q' to quit.")

    # ----------------------------------------------------------------------------
    # 5. FRAME-BY-FRAME TRACKING
    # ----------------------------------------------------------------------------
    while cap.isOpened():  # process every frame until the video ends
        success, frame = cap.read()  # read the next frame

        if not success:  # exit loop if frame cannot be read or video has ended
            print("Video ended or frame could not be read.")
            break

        results = model.track(
            frame,                       # input image frame
            persist=True,                # keeps the same ID assigned to the same object across frames
            conf=0.3,                    # minimum confidence score threshold (0-1)
            iou=0.5,                     # box overlap threshold (Intersection over Union)
            tracker="bytetrack.yaml"     # ByteTrack algorithm configuration
        )

        annotated_frame = results[0].plot(font_size=0.6, line_width=1)  # draw boxes and IDs on frame

        cv2.imshow("YOLOv8 Tracking", annotated_frame)  # display the annotated frame
        out.write(annotated_frame)                       # write the annotated frame to output video

        if cv2.waitKey(1) & 0xFF == ord("q"):  # stop tracking if 'q' is pressed
            print("Stopped by user.")
            break

# ----------------------------------------------------------------------------
# 6. EXCEPTION HANDLING
# ----------------------------------------------------------------------------
except FileNotFoundError as e:
    print(f"\nFile Error: {e}")

except KeyboardInterrupt:
    print("\nProgram was forcefully closed by the user.")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

# ----------------------------------------------------------------------------
# 7. CLEANUP
# ----------------------------------------------------------------------------
finally:
    print("\nReleasing resources...")

    if cap is not None:
        cap.release()              # release video reader
    if out is not None:
        out.release()              # release video writer
    cv2.destroyAllWindows()        # close all open windows

    print("Program terminated.")