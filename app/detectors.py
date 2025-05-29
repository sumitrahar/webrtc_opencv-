import torch
from ultralytics import YOLO
import datetime
import os
from collections import deque
import numpy as np

class OptimizedObjectDetector:
    """Optimized object detection system for monitoring applications"""
    
    def __init__(self, 
                 model_name='yolov12n.pt',
                 confidence_threshold=0.4,
                 alert_window_seconds=3,
                 fps_estimate=10,
                 alert_frames_threshold_ratio=0.6,
                 monitored_classes=None):
        """
        Initialize the object detection system
        
        Args:
            model_name: YOLO model to use
            confidence_threshold: Minimum confidence for detections
            alert_window_seconds: Time window for alert analysis
            fps_estimate: Estimated FPS for alert calculations
            alert_frames_threshold_ratio: Ratio of frames needed to trigger alert
            monitored_classes: Dict of classes to monitor {'class_name': 'alert_message'}
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.alert_window_seconds = alert_window_seconds
        self.fps_estimate = fps_estimate
        self.alert_frames_threshold_ratio = alert_frames_threshold_ratio
        
        # Default monitored classes
        if monitored_classes is None:
            self.monitored_classes = {
                'cell phone': 'Phone Detected',
                'person': 'Multiple People',
                'book': 'Book Detected'
            }
        else:
            self.monitored_classes = monitored_classes
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.setup_model()
        self.setup_alert_system()
        self.setup_logging()
        
    def setup_model(self):
        """Initialize YOLO model"""
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            
            # Filter valid classes that exist in the model
            self.valid_classes = {k: v for k, v in self.monitored_classes.items() 
                                if k in self.model.names.values()}
            
            print(f"YOLO loaded on {self.device}")
            print(f"Monitoring classes: {list(self.valid_classes.keys())}")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            self.valid_classes = {}
    
    def setup_alert_system(self):
        """Initialize alert tracking system"""
        max_frames = int(self.fps_estimate * self.alert_window_seconds)
        self.recent_detections = {alert: deque(maxlen=max_frames) 
                                for alert in self.valid_classes.values()}
        self.previous_alerts = set()
        
    def setup_logging(self):
        """Initialize logging system"""
        log_dir = "detection_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(log_dir, f"detection_log_{timestamp}.txt")
        
        with open(self.log_filename, 'w') as f:
            f.write(f"OBJECT DETECTION SESSION - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_name} | Device: {self.device} | Confidence: {self.confidence_threshold}\n")
            f.write(f"Monitored Classes: {list(self.valid_classes.keys())}\n\n")
        
        print(f"Logging to: {self.log_filename}")
    
    def detect_objects(self, frame):
        """
        Perform object detection on a frame
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            tuple: (annotated_frame, detected_objects, alerts)
                - annotated_frame: Frame with detection boxes drawn
                - detected_objects: List of detected object strings
                - alerts: List of current alert messages
        """
        if self.model is None:
            return frame, [], []
        
        try:
            # Run YOLO inference
            results = self.model.predict(
                source=frame, 
                device=self.device, 
                verbose=False, 
                conf=self.confidence_threshold
            )[0]
            
            # Get annotated frame with bounding boxes
            annotated_frame = results.plot(labels=True, conf=True)
            
            # Process detections
            detected_objects, alerts = self._process_detections(results)
            
            return annotated_frame, detected_objects, alerts
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, [], []
    
    def _process_detections(self, results):
        """Process YOLO results and generate alerts"""
        current_detections = {alert: False for alert in self.valid_classes.values()}
        person_count = 0
        detected_objects = []
        
        if results.boxes is not None:
            for box in results.boxes:
                class_name = self.model.names[int(box.cls.item())]
                confidence = float(box.conf.item())
                detected_objects.append(f"{class_name} ({confidence:.2f})")
                
                # Count persons separately for multiple people detection
                if class_name == 'person':
                    person_count += 1
                elif class_name in self.valid_classes:
                    current_detections[self.valid_classes[class_name]] = True
        
        # Handle multiple people detection
        if 'Multiple People' in current_detections:
            current_detections['Multiple People'] = person_count > 1
        
        # Update detection history and generate alerts
        alerts = []
        for alert_name, detection_deque in self.recent_detections.items():
            detection_deque.append(current_detections[alert_name])
            
            # Check if we have enough frames to make a decision
            if len(detection_deque) == detection_deque.maxlen:
                alert_count = sum(detection_deque)
                threshold = detection_deque.maxlen * self.alert_frames_threshold_ratio
                
                if alert_count >= threshold:
                    alert_msg = f"ALERT: {alert_name}"
                    alerts.append(alert_msg)
        
        # Log new alerts
        self._log_new_alerts(alerts)
        
        return detected_objects, alerts
    
    def _log_new_alerts(self, current_alerts):
        """Log new alerts to file"""
        current_alert_set = set(current_alerts)
        new_alerts = current_alert_set - self.previous_alerts
        
        if new_alerts:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.log_filename, 'a') as f:
                for alert in new_alerts:
                    f.write(f"[{timestamp}] *** {alert} ***\n")
        
        self.previous_alerts = current_alert_set
    
    def log_detection_info(self, fps, detected_objects, alerts):
        """Log detection information"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_filename, 'a') as f:
            f.write(f"[{timestamp}] FPS: {fps:.1f}\n")
            f.write(f"  Objects: {', '.join(detected_objects) if detected_objects else 'None'}\n")
            f.write(f"  Alerts: {', '.join(alerts) if alerts else 'None'}\n")
    
    def get_detection_stats(self):
        """Get current detection statistics"""
        stats = {}
        for alert_name, detection_deque in self.recent_detections.items():
            if len(detection_deque) > 0:
                detection_rate = sum(detection_deque) / len(detection_deque)
                stats[alert_name] = {
                    'detection_rate': detection_rate,
                    'recent_frames': len(detection_deque),
                    'positive_detections': sum(detection_deque)
                }
        return stats
    
    def reset_alert_history(self):
        """Reset alert history (useful for new sessions)"""
        for detection_deque in self.recent_detections.values():
            detection_deque.clear()
        self.previous_alerts = set()
    
    def update_confidence_threshold(self, new_threshold):
        """Update confidence threshold"""
        self.confidence_threshold = new_threshold
        print(f"Confidence threshold updated to: {new_threshold}")
    
    def add_monitored_class(self, class_name, alert_message):
        """Add a new class to monitor"""
        if class_name in self.model.names.values():
            self.valid_classes[class_name] = alert_message
            max_frames = int(self.fps_estimate * self.alert_window_seconds)
            self.recent_detections[alert_message] = deque(maxlen=max_frames)
            print(f"Added monitoring for: {class_name} -> {alert_message}")
        else:
            print(f"Class '{class_name}' not found in model")
    
    def close(self):
        """Clean up resources and close log file"""
        if hasattr(self, 'log_filename'):
            with open(self.log_filename, 'a') as f:
                f.write(f"\nDetection session ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"Detection logs saved to: {self.log_filename}")


# Example usage function
def example_usage():
    """Example of how to integrate the detector into your app"""
    import cv2
    import time
    
    # Initialize detector with custom settings
    detector = OptimizedObjectDetector(
        confidence_threshold=0.5,
        monitored_classes={
            'cell phone': 'Phone Detected',
            'person': 'Multiple People',
            'laptop': 'Laptop Detected',
            'book': 'Book Detected'
        }
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    prev_time = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            
            # Perform detection
            annotated_frame, detected_objects, alerts = detector.detect_objects(frame)
            
            # Add FPS to display
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display alerts
            for i, alert in enumerate(alerts):
                cv2.putText(annotated_frame, alert, (10, 60 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Object Detection', annotated_frame)
            
            # Log periodically (every second)
            if int(current_time) % 1 == 0:
                detector.log_detection_info(fps, detected_objects, alerts)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

if __name__ == "__main__":
    example_usage()