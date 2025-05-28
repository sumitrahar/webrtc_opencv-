import json
import asyncio
import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder
import os
import time
from collections import deque
import logging
from datetime import datetime
import sys

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Setup main logger
main_log_filename = f"logs/test_object_detection_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
main_logger = logging.getLogger('main')
main_logger.setLevel(logging.INFO)
main_handler = logging.FileHandler(main_log_filename, mode='w')
main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
main_logger.addHandler(main_handler)
main_logger.addHandler(logging.StreamHandler(sys.stdout))

# Setup detection logger
detection_log_filename = f"logs/test_object_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
detection_logger = logging.getLogger('detection')
detection_logger.setLevel(logging.INFO)
detection_handler = logging.FileHandler(detection_log_filename, mode='w')
detection_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
detection_logger.addHandler(detection_handler)
detection_logger.addHandler(logging.StreamHandler(sys.stdout))

# YOLO Detection Configuration
try:
    import torch
    from ultralytics import YOLO
    main_logger.info("Successfully imported torch and YOLO")
    detection_logger.info("Successfully imported torch and YOLO")
    YOLO_AVAILABLE = True
except ImportError as e:
    main_logger.error(f"Error importing YOLO dependencies: {e}")
    detection_logger.error(f"Error importing YOLO dependencies: {e}")
    print(f"Error importing YOLO dependencies: {e}")
    print("Please install required packages: pip install ultralytics torch torchvision opencv-python")
    YOLO_AVAILABLE = False

MODEL_NAME = 'yolov8n.pt'  # Using the smallest YOLOv8 model for better performance
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
CLASSES_OF_INTEREST = {
    'cell phone': 'Phone Detected',
    'person': 'Another Person Present',
    'book': 'Book Detected',
    'laptop': 'Laptop Detected',
    'mouse': 'Mouse Detected'
}
CONFIDENCE_THRESHOLD = 0.5
ALERT_WINDOW_SECONDS = 3
FPS_ESTIMATE = 5
ALERT_FRAMES_THRESHOLD_RATIO = 0.6

class ObjectDetectionTrack:
    def __init__(self, track):
        self.track = track
        self.model = None
        self.recent_detections = {}
        self.max_frames_in_window = 0
        self.current_classes_of_interest = {}
        self.alert_messages = []
        self.frame_count = 0
        self.detection_log_file = None

        # Create a detailed log file for frame-by-frame detections
        self.detection_log_filename = f"logs/test_detection_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            self.detection_log_file = open(self.detection_log_filename, 'w')
            self.detection_log_file.write(f"=== Object Detection Test Log ===\n")
            self.detection_log_file.write(f"Started at: {datetime.now()}\n")
            self.detection_log_file.write(f"Model: {MODEL_NAME}\n")
            self.detection_log_file.write(f"Device: {DEVICE}\n")
            self.detection_log_file.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
            self.detection_log_file.write("="*50 + "\n\n")
            self.detection_log_file.flush()
            main_logger.info(f"Created detection log file: {self.detection_log_filename}")
            detection_logger.info(f"Created detection log file: {self.detection_log_filename}")
        except Exception as e:
            main_logger.error(f"Failed to create detection log file: {e}")
            detection_logger.error(f"Failed to create detection log file: {e}")

        if YOLO_AVAILABLE:
            try:
                self.initialize_yolo()
                main_logger.info("ObjectDetectionTrack initialized with YOLO model")
                detection_logger.info("ObjectDetectionTrack initialized with YOLO model")
            except Exception as e:
                main_logger.error(f"Failed to initialize YOLO model: {e}", exc_info=True)
                detection_logger.error(f"Failed to initialize YOLO model: {e}", exc_info=True)
                YOLO_AVAILABLE = False
        else:
            main_logger.warning("ObjectDetectionTrack initialized without YOLO model - detection will not work")
            detection_logger.warning("ObjectDetectionTrack initialized without YOLO model - detection will not work")

    def initialize_yolo(self):
        try:
            main_logger.info(f"Loading YOLO model: {MODEL_NAME} on device: {DEVICE}")
            detection_logger.info(f"Loading YOLO model: {MODEL_NAME} on device: {DEVICE}")
            
            # Check if model file exists
            if not os.path.exists(MODEL_NAME):
                main_logger.info(f"Downloading YOLO model: {MODEL_NAME}")
                detection_logger.info(f"Downloading YOLO model: {MODEL_NAME}")
                self.model = YOLO(MODEL_NAME)  # This will download the model if not present
            else:
                self.model = YOLO(MODEL_NAME)

            self.model.to(DEVICE)

            if not hasattr(self.model, 'names'):
                raise RuntimeError("YOLO model not properly loaded - missing class names")

            coco_class_names = self.model.names
            main_logger.info(f"Model loaded successfully. COCO classes: {len(coco_class_names)}")
            detection_logger.info(f"Model loaded successfully. COCO classes: {len(coco_class_names)}")

            valid_classes_of_interest = {}
            for class_name, alert_name in CLASSES_OF_INTEREST.items():
                if class_name in coco_class_names.values():
                    valid_classes_of_interest[class_name] = alert_name
                    main_logger.info(f"Valid class found: {class_name} -> {alert_name}")
                    detection_logger.info(f"Valid class found: {class_name} -> {alert_name}")
                else:
                    main_logger.warning(f"Class '{class_name}' not found in model classes")
                    detection_logger.warning(f"Class '{class_name}' not found in model classes")

            if not valid_classes_of_interest:
                main_logger.error("No valid classes found in CLASSES_OF_INTEREST")
                detection_logger.error("No valid classes found in CLASSES_OF_INTEREST")
                self.model = None
                return

            self.current_classes_of_interest = valid_classes_of_interest
            self.max_frames_in_window = int(FPS_ESTIMATE * ALERT_WINDOW_SECONDS)
            self.recent_detections = {
                alert_name: deque(maxlen=self.max_frames_in_window)
                for alert_name in self.current_classes_of_interest.values()
            }

        except Exception as e:
            main_logger.error(f"Error initializing YOLO model: {e}", exc_info=True)
            detection_logger.error(f"Error initializing YOLO model: {e}", exc_info=True)
            self.model = None
            raise  # Re-raise the exception to be caught by the caller

    async def recv(self):
        frame = await self.track.recv()

        if self.model and hasattr(frame, 'to_ndarray'):
            try:
                img = frame.to_ndarray(format="bgr24")
                await asyncio.to_thread(self.process_frame_detection_sync, img)
            except Exception as e:
                main_logger.error(f"Error processing frame: {e}", exc_info=True)
                detection_logger.error(f"Error processing frame: {e}", exc_info=True)

        return frame

    def process_frame_detection_sync(self, frame_image):
        try:
            self.frame_count += 1

            if frame_image is None or frame_image.size == 0:
                main_logger.warning(f"Invalid frame received at count {self.frame_count}")
                detection_logger.warning(f"Invalid frame received at count {self.frame_count}")
                return

            # Log frame processing
            if self.frame_count % 30 == 0:  # Log every 30 frames
                main_logger.info(f"Processing frame {self.frame_count}")
                detection_logger.info(f"Processing frame {self.frame_count}")

            results = self.model.predict(
                source=frame_image,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD,
                device=DEVICE
            )
            result = results[0]

            current_frame_detected_alerts = {
                alert_name: False
                for alert_name in self.current_classes_of_interest.values()
            }
            num_persons_detected = 0
            frame_detections_info = []

            if result.boxes is not None and len(result.boxes) > 0:
                coco_class_names = self.model.names

                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name_from_model = coco_class_names[cls_id]

                    if class_name_from_model == 'person':
                        num_persons_detected += 1

                    if class_name_from_model in self.current_classes_of_interest:
                        alert_name = self.current_classes_of_interest[class_name_from_model]
                        if alert_name != 'Another Person Present':
                            current_frame_detected_alerts[alert_name] = True
                            detection_info = f"{class_name_from_model} (confidence: {confidence:.2f})"
                            frame_detections_info.append(detection_info)
                            detection_logger.info(f"Frame {self.frame_count}: {detection_info}")

            # Handle "Another Person Present" alert
            alert_name_another_person = 'Another Person Present'
            if alert_name_another_person in self.current_classes_of_interest.values():
                is_another_person_present = num_persons_detected > 1
                current_frame_detected_alerts[alert_name_another_person] = is_another_person_present
                if is_another_person_present:
                    detection_info = f"Multiple persons ({num_persons_detected})"
                    frame_detections_info.append(detection_info)
                    detection_logger.info(f"Frame {self.frame_count}: {detection_info}")

            # Write to detailed log file
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_entry = f"[{timestamp}] Frame {self.frame_count}: "
            
            if frame_detections_info:
                log_entry += f"Detected: {', '.join(frame_detections_info)}"
                main_logger.info(log_entry)
                detection_logger.info(log_entry)
            elif self.frame_count % (FPS_ESTIMATE * 5) == 0:
                log_entry += "No objects of interest detected"
                main_logger.info(log_entry)
                detection_logger.info(log_entry)
            
            if self.detection_log_file and not self.detection_log_file.closed:
                self.detection_log_file.write(log_entry + "\n")
                self.detection_log_file.flush()

            # Update detection history and check for sustained alerts
            for alert_name, deque_obj in self.recent_detections.items():
                deque_obj.append(current_frame_detected_alerts.get(alert_name, False))

            self.alert_messages.clear()
            for alert_name, deque_obj in self.recent_detections.items():
                if len(deque_obj) == self.max_frames_in_window:
                    true_counts = sum(deque_obj)
                    if true_counts >= self.max_frames_in_window * ALERT_FRAMES_THRESHOLD_RATIO:
                        alert_msg = f"SUSTAINED ALERT: {alert_name}"
                        self.alert_messages.append(alert_msg)
                        main_logger.warning(f"Frame {self.frame_count}: {alert_msg}")
                        detection_logger.warning(f"Frame {self.frame_count}: {alert_msg}")
                        
                        if self.detection_log_file and not self.detection_log_file.closed:
                            alert_log = f"[{timestamp}] *** {alert_msg} *** (ratio: {true_counts}/{self.max_frames_in_window})"
                            self.detection_log_file.write(alert_log + "\n")
                            self.detection_log_file.flush()

        except Exception as e:
            error_msg = f"Error in frame detection processing (frame {self.frame_count}): {e}"
            main_logger.error(error_msg, exc_info=True)
            detection_logger.error(error_msg, exc_info=True)
            if self.detection_log_file and not self.detection_log_file.closed:
                self.detection_log_file.write(f"[{datetime.now()}] ERROR: {error_msg}\n")
                self.detection_log_file.flush()

    def __del__(self):
        if hasattr(self, 'detection_log_file') and self.detection_log_file:
            try:
                if not self.detection_log_file.closed:
                    end_message = f"\nTest session ended at {datetime.now()}\n"
                    self.detection_log_file.write(end_message)
                    main_logger.info(end_message.strip())
                    detection_logger.info(end_message.strip())
                    self.detection_log_file.close()
            except Exception as e:
                main_logger.error(f"Error closing detection_log_file: {e}")
                detection_logger.error(f"Error closing detection_log_file: {e}")


class WebRTCConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.pc = RTCPeerConnection()
        self.recorder = None
        self.detection_track = None # Keep a reference for cleanup
        main_logger.info("WebRTC connection established for test session")

    async def disconnect(self, close_code):
        main_logger.info(f"WebRTC test session closing/closed with code: {close_code}")
        if self.recorder:
            main_logger.info("Stopping recorder...")
            await self.recorder.stop()
            main_logger.info("Recorder stopped.")
        
        # Clean up ObjectDetectionTrack instance to ensure __del__ is called for log file closing
        if hasattr(self, 'detection_track') and self.detection_track:
            main_logger.info("Cleaning up object detection track...")
            # Explicitly call __del__ or a custom cleanup method if __del__ is unreliable
            # For simplicity, relying on garbage collection here, but manual cleanup is safer
            del self.detection_track 
            self.detection_track = None
            main_logger.info("Object detection track cleaned up.")

        if self.pc and self.pc.signalingState != "closed":
            main_logger.info("Closing PeerConnection...")
            await self.pc.close()
            main_logger.info("PeerConnection closed.")
        
        main_logger.info("WebRTC disconnect process complete.")


    async def receive(self, text_data):
        data = json.loads(text_data)
        
        if data['type'] == 'offer':
            try:
                offer = RTCSessionDescription(sdp=data['sdp'], type='offer')
                await self.pc.setRemoteDescription(offer)
                
                answer = await self.pc.createAnswer()
                await self.pc.setLocalDescription(answer)
                
                await self.send(text_data=json.dumps({
                    'type': 'answer',
                    'sdp': self.pc.localDescription.sdp # Use sdp from localDescription
                }))
                
                @self.pc.on('track')
                async def on_track(track):
                    main_logger.info(f"Received {track.kind} track for test")
                    
                    if track.kind == "video":
                        os.makedirs("recordings", exist_ok=True)
                        
                        test_recording_name = f"recordings/test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        
                        if YOLO_AVAILABLE and self.pc: # Ensure pc is available
                            main_logger.info("YOLO available, wrapping track for object detection.")
                            self.detection_track = ObjectDetectionTrack(track) # Store reference
                            self.recorder = MediaRecorder(test_recording_name)
                            self.recorder.addTrack(self.detection_track)
                            main_logger.info(f"Test object detection enabled. Recording to {test_recording_name}")
                        else:
                            main_logger.warning(f"YOLO not available or PC not ready. Recording test video without detection to {test_recording_name}")
                            self.recorder = MediaRecorder(test_recording_name)
                            self.recorder.addTrack(track)
                        
                        if self.recorder:
                            await self.recorder.start()
                            main_logger.info("Test recording started")
                    
                    @track.on('ended')
                    async def on_ended():
                        main_logger.info(f"Track {track.kind} ended")
                        if self.recorder:
                            main_logger.info("Track ended, stopping recorder.")
                            await self.recorder.stop()
                            main_logger.info("Recorder stopped due to track end.")

            except Exception as e:
                main_logger.error(f"Error handling offer or track: {e}", exc_info=True)
                # Optionally send an error back to the client if appropriate

        elif data['type'] == 'candidate':
            try:
                candidate_data = data['candidate']
                # Make sure candidate_data is not None and has the required fields
                if candidate_data and 'candidate' in candidate_data and \
                   'sdpMid' in candidate_data and 'sdpMLineIndex' in candidate_data:
                    candidate = aiortc.RTCIceCandidate(
                        candidate_data['candidate'],
                        candidate_data['sdpMid'],
                        candidate_data['sdpMLineIndex']
                    )
                    await self.pc.addIceCandidate(candidate)
                elif candidate_data is None: # Handle null candidate signaling end of candidates
                    main_logger.info("Received null ICE candidate, signaling end of candidates.")
                    await self.pc.addIceCandidate(None)
                else:
                    main_logger.warning(f"Received malformed ICE candidate data: {candidate_data}")
            except Exception as e:
                main_logger.error(f"Error handling ICE candidate: {e}", exc_info=True)