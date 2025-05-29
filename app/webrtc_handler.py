# app/webrtc_handler.py

from fastapi import APIRouter, WebSocket
import cv2
import numpy as np
import base64
from app.detectors import detect_cheating

router = APIRouter()

@router.websocket("/ws/video")
async def websocket_video_stream(websocket : WebSocket):
    await websocket.accept()
    print("Client connected to /ws/video")

    try:
        while True:
            # Receive base64-encoded frame from the frontend
            data = await websocket.receive_text()

            # Convert base64 to NumPy array (OpenCV image)
            image_data = base64.b64decode(data.split(',')[1])
            np_arr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                # Run YOLO + head pose detection
                annotated_frame = detect_cheating(frame)

                # Optional : you can send annotated frame back (base64-encoded) for feedback
                # encoded, buffer = cv2.imencode('.jpg', annotated_frame)
                # b64_frame = base64.b64encode(buffer).decode('utf-8)
                # await websocket.send_text(f"data:image/jpeg;base64,{b64_frame}")
            else:
                print("Invalid frame received")

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        print("WebSocket connection closed")