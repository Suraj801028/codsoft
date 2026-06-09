import cv2
import numpy as np

def draw_face_annotations(frame, face_results, draw_landmarks=True):
    """
    Draws stylized bounding boxes, labels, and landmarks on a frame.
    face_results: list of dicts from FacePipeline.process_and_recognize
    """
    annotated = frame.copy()
    
    for res in face_results:
        x, y, w, h = res['box']
        name = res['name']
        similarity = res['similarity']
        confidence = res['confidence']
        landmarks = res['landmarks']
        
        # Color palette: Green for recognized, Red for Unknown, Cyan for details
        is_recognized = name != "Unknown"
        color = (46, 204, 113) if is_recognized else (231, 76, 60)  # Green vs Red (BGR)
        text_color = (255, 255, 255)
        
        # 1. Draw modern bounding box with thick corners
        thickness = 2
        corner_len = min(20, w // 4, h // 4)
        
        # Draw main rectangle slightly thin
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)
        
        # Draw thick corners
        # Top-Left
        cv2.line(annotated, (x, y), (x + corner_len, y), color, thickness + 1)
        cv2.line(annotated, (x, y), (x, y + corner_len), color, thickness + 1)
        # Top-Right
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color, thickness + 1)
        cv2.line(annotated, (x + w, y), (x + w, y + corner_len), color, thickness + 1)
        # Bottom-Left
        cv2.line(annotated, (x, y + h), (x + corner_len, y + h), color, thickness + 1)
        cv2.line(annotated, (x, y + h), (x, y + h - corner_len), color, thickness + 1)
        # Bottom-Right
        cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), color, thickness + 1)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), color, thickness + 1)
        
        # 2. Draw landmarks if available
        if draw_landmarks and landmarks:
            for (lx, ly) in landmarks:
                cv2.circle(annotated, (lx, ly), 3, (0, 255, 255), -1)  # Yellow dots for landmarks
                
        # 3. Draw text label background card
        label = f"{name}"
        if is_recognized:
            label += f" ({similarity:.2f})"
        else:
            # Show face detector confidence for unknown faces
            label += f" (Det: {confidence:.2f})"
            
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, 1)
        
        # Label card position (above the box or inside the box if top edge is out of frame)
        card_y1 = max(lh + 6, y - 6) - lh - 6
        card_y2 = max(lh + 6, y - 6) + 2
        card_x1 = x
        card_x2 = x + lw + 12
        
        # Clip card size inside frame boundary
        h_frame, w_frame = frame.shape[:2]
        card_x2 = min(w_frame - 1, card_x2)
        
        # Draw background filled card with a drop shadow or solid matching color
        cv2.rectangle(annotated, (card_x1, card_y1), (card_x2, card_y2), color, cv2.FILLED)
        cv2.putText(annotated, label, (card_x1 + 6, card_y2 - 5), font, font_scale, text_color, 1, cv2.LINE_AA)
        
    return annotated

def process_video_file(pipeline, input_path, output_path, detector_name, recognizer_name, 
                       conf_thresh, sim_thresh, progress_callback=None):
    """
    Reads an input video, runs face detection and recognition, writes annotated frames to output_path.
    progress_callback is called with (current_frame, total_frames).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            results = pipeline.process_and_recognize(
                frame,
                detector_name=detector_name,
                recognizer_name=recognizer_name,
                conf_threshold=conf_thresh,
                similarity_threshold=sim_thresh
            )
            
            # Annotate frame
            annotated_frame = draw_face_annotations(frame, results)
            
            # Write frame
            out.write(annotated_frame)
            
            frame_count += 1
            if progress_callback:
                # Call callback, check if it wants to cancel (if it returns False)
                if progress_callback(frame_count, total_frames) is False:
                    break
    finally:
        cap.release()
        out.release()
        
    return True
