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

# YOLO Detection Configuration
try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO dependencies not available. Install ultralytics and torch for object detection.")

MODEL_NAME = 'yolov8m.pt'  # Fixed: Changed from yolov12m.pt to yolov8m.pt
DEVICE = 'cpu'  # Start with CPU for testing
# Fixed: Use class names that match COCO dataset exactly
CLASSES_OF_INTEREST = {
    'cell phone': 'Phone Detected',
    'person': 'Another Person Present', 
    'book': 'Book Detected'
}
CONFIDENCE_THRESHOLD = 0.4
ALERT_WINDOW_SECONDS = 3
FPS_ESTIMATE = 5
ALERT_FRAMES_THRESHOLD_RATIO = 0.6

# Setup logging with test filename
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/test_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ObjectDetectionTrack:
    """Custom track wrapper for object detection"""
    
    def __init__(self, track):
        self.track = track
        self.model = None
        self.recent_detections = {}
        self.max_frames_in_window = 0
        self.current_classes_of_interest = {}
        self.alert_messages = []
        self.frame_count = 0
        self.detection_log_file = None
        
        # Create test detection log file
        self.detection_log_filename = f"logs/test_object_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        if YOLO_AVAILABLE:
            self.initialize_yolo()
    
    def initialize_yolo(self):
        """Initialize YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {MODEL_NAME} on device: {DEVICE}")
            self.model = YOLO(MODEL_NAME)
            
            # Get COCO class names (this returns a dict with id->name mapping)
            coco_class_names = self.model.names
            logger.info(f"Model loaded successfully. COCO classes: {len(coco_class_names)}")
            
            # Create detection log file
            self.detection_log_file = open(self.detection_log_filename, 'w')
            self.detection_log_file.write(f"TEST OBJECT DETECTION LOG - Started at {datetime.now()}\n")
            self.detection_log_file.write("="*50 + "\n")
            self.detection_log_file.flush()
            
            # Fixed: Validate classes of interest against COCO class names
            valid_classes_of_interest = {}
            for class_name, alert_name in CLASSES_OF_INTEREST.items():
                # Check if class name exists in COCO class names (values of the dict)
                if class_name in coco_class_names.values():
                    valid_classes_of_interest[class_name] = alert_name
                    logger.info(f"Valid class found: {class_name} -> {alert_name}")
                else:
                    logger.warning(f"Class '{class_name}' not found in model classes")
                    # Print available classes for debugging
                    logger.info(f"Available classes: {list(coco_class_names.values())}")
            
            if not valid_classes_of_interest:
                logger.error("No valid classes found in CLASSES_OF_INTEREST")
                return
                
            self.current_classes_of_interest = valid_classes_of_interest
            logger.info(f"Monitoring for: {self.current_classes_of_interest}")
            
            # Log to test file
            self.detection_log_file.write(f"Monitoring classes: {self.current_classes_of_interest}\n")
            self.detection_log_file.write(f"Confidence threshold: {CONFIDENCE_THRESHOLD}\n")
            self.detection_log_file.write("-"*50 + "\n")
            self.detection_log_file.flush()
            
            # Initialize detection tracking
            self.max_frames_in_window = int(FPS_ESTIMATE * ALERT_WINDOW_SECONDS)
            self.recent_detections = {
                alert_name: deque(maxlen=self.max_frames_in_window)
                for alert_name in self.current_classes_of_interest.values()
            }
            
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            self.model = None
    
    async def recv(self):
        """Override recv to process frames for object detection"""
        frame = await self.track.recv()
        
        if self.model and hasattr(frame, 'to_ndarray'):
            try:
                # Convert frame to numpy array
                img = frame.to_ndarray(format="bgr24")
                await self.process_frame_detection(img)
            except Exception as e:
                logger.error(f"Error processing frame for detection: {e}")
        
        return frame
    
    async def process_frame_detection(self, frame):
        """Process frame for object detection"""
        try:
            self.frame_count += 1
            
            # Run YOLO inference
            results = self.model.predict(
                source=frame, 
                verbose=False, 
                conf=CONFIDENCE_THRESHOLD
            )
            result = results[0]
            
            # Process detections
            current_frame_detected_alerts = {
                alert_name: False 
                for alert_name in self.current_classes_of_interest.values()
            }
            num_persons_detected = 0
            frame_detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                coco_class_names = self.model.names
                
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    cls_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name_from_model = coco_class_names[cls_id]  # Get class name from ID
                    
                    # Count persons
                    if class_name_from_model == 'person':
                        num_persons_detected += 1
                    
                    # Check if this class is one we're monitoring
                    if class_name_from_model in self.current_classes_of_interest:
                        alert_name = self.current_classes_of_interest[class_name_from_model]
                        if alert_name != 'Another Person Present':
                            current_frame_detected_alerts[alert_name] = True
                            detection_info = f"{class_name_from_model} (confidence: {confidence:.2f})"
                            frame_detections.append(detection_info)
                            logger.info(f"Frame {self.frame_count}: Detected {detection_info}")
            
            # Handle multiple person detection
            if 'Another Person Present' in self.current_classes_of_interest.values():
                current_frame_detected_alerts['Another Person Present'] = num_persons_detected > 1
                if num_persons_detected > 1:
                    detection_info = f"Multiple persons ({num_persons_detected})"
                    frame_detections.append(detection_info)
                    logger.info(f"Frame {self.frame_count}: {detection_info}")
            
            # Log detections to test file
            if self.detection_log_file:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                if frame_detections:
                    self.detection_log_file.write(f"[{timestamp}] Frame {self.frame_count}: {', '.join(frame_detections)}\n")
                else:
                    # Log every 30th frame even if no detections for debugging
                    if self.frame_count % 30 == 0:
                        self.detection_log_file.write(f"[{timestamp}] Frame {self.frame_count}: No detections\n")
                self.detection_log_file.flush()
            
            # Update recent detections
            for alert_name, deque_obj in self.recent_detections.items():
                deque_obj.append(current_frame_detected_alerts[alert_name])
            
            # Check for sustained alerts
            self.alert_messages.clear()
            for alert_name, deque_obj in self.recent_detections.items():
                if len(deque_obj) == self.max_frames_in_window:
                    true_counts = sum(deque_obj)
                    if true_counts >= self.max_frames_in_window * ALERT_FRAMES_THRESHOLD_RATIO:
                        alert_msg = f"SUSTAINED ALERT: {alert_name}"
                        self.alert_messages.append(alert_msg)
                        logger.warning(f"Frame {self.frame_count}: {alert_msg} (detection ratio: {true_counts}/{self.max_frames_in_window})")
                        
                        # Log sustained alert to test file
                        if self.detection_log_file:
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            self.detection_log_file.write(f"[{timestamp}] *** {alert_msg} *** (ratio: {true_counts}/{self.max_frames_in_window})\n")
                            self.detection_log_file.flush()
            
        except Exception as e:
            logger.error(f"Error in frame detection processing: {e}")
            # Log error to test file too
            if self.detection_log_file:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                self.detection_log_file.write(f"[{timestamp}] ERROR: {str(e)}\n")
                self.detection_log_file.flush()
    
    def __del__(self):
        """Clean up log file"""
        if hasattr(self, 'detection_log_file') and self.detection_log_file:
            self.detection_log_file.write(f"\nTest session ended at {datetime.now()}\n")
            self.detection_log_file.close()

class WebRTCConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.pc = RTCPeerConnection()
        self.recorder = None
        logger.info("WebRTC connection established for test session")

    async def disconnect(self, close_code):
        if self.recorder:
            await self.recorder.stop()
        await self.pc.close()
        logger.info(f"WebRTC test session closed with code: {close_code}")

    async def receive(self, text_data):
        data = json.loads(text_data)
        
        if data['type'] == 'offer':
            # Handle the offer
            offer = RTCSessionDescription(sdp=data['sdp'], type='offer')
            await self.pc.setRemoteDescription(offer)
            
            # Create answer
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            
            # Send answer back
            await self.send(text_data=json.dumps({
                'type': 'answer',
                'sdp': answer.sdp
            }))
            
            # Setup media handlers
            @self.pc.on('track')
            async def on_track(track):
                logger.info(f"Received {track.kind} track for test")
                
                if track.kind == "video":
                    # Ensure directory exists
                    os.makedirs("recordings", exist_ok=True)
                    os.makedirs("logs", exist_ok=True)
                    
                    # Create test recording filename
                    test_recording_name = f"recordings/test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    
                    # Wrap track with object detection
                    if YOLO_AVAILABLE:
                        detection_track = ObjectDetectionTrack(track)
                        self.recorder = MediaRecorder(test_recording_name)
                        self.recorder.addTrack(detection_track)
                        logger.info(f"Test object detection enabled for video track - Recording to {test_recording_name}")
                    else:
                        self.recorder = MediaRecorder(test_recording_name)
                        self.recorder.addTrack(track)
                        logger.warning(f"Object detection not available - recording test video without detection to {test_recording_name}")
                    
                    await self.recorder.start()
                    logger.info("Test recording started")

        elif data['type'] == 'candidate':
            await self.pc.addIceCandidate(aiortc.RTCIceCandidate(
                data['candidate']['candidate'],
                data['candidate']['sdpMid'],
                data['candidate']['sdpMLineIndex']
            ))