"""
Generated with the help of GPT-4.1 from face_mesh.py and obj_tracker.py
"""
# DepthAI + MediaPipe Face Mesh 3D mapping
import cv2
import depthai as dai
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Create DepthAI pipeline for RGB and depth (correct API)
pipeline = dai.Pipeline()

# RGB camera
cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
rgb_output = cam_rgb.requestOutput((640, 480))
rgb_queue = rgb_output.createOutputQueue()

# Mono cameras for stereo depth
mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
left_output = mono_left.requestOutput((640, 400))
right_output = mono_right.requestOutput((640, 400))
left_queue = left_output.createOutputQueue()
right_queue = right_output.createOutputQueue()

# Stereo depth node
stereo = pipeline.create(dai.node.StereoDepth)
left_output.link(stereo.left)
right_output.link(stereo.right)
depth_output = stereo.depth
depth_queue = depth_output.createOutputQueue()

# Start pipeline
with pipeline:
  pipeline.start()
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while pipeline.isRunning():
      in_rgb = rgb_queue.get()
      in_depth = depth_queue.get()
      frame = in_rgb.getCvFrame()
      depth_frame = in_depth.getFrame() # 16-bit depth map
      # Resize depth frame to match RGB frame size
      depth_frame_resized = cv2.resize(depth_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

      # MediaPipe expects RGB
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(rgb_frame)

      # Draw face mesh and get 3D coordinates
      frame_out = frame.copy()
      if results.multi_face_landmarks:
        # Key landmark indices and their labels
        landmark_labels = {
          1: "Nose Tip",
          33: "Left Eye",
          263: "Right Eye",
          61: "Mouth"
        }
        for face_landmarks in results.multi_face_landmarks:
          mp_drawing.draw_landmarks(
            image=frame_out,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
          mp_drawing.draw_landmarks(
            image=frame_out,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
          mp_drawing.draw_landmarks(
            image=frame_out,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

          # Get 3D coordinates for each landmark
          h, w, _ = frame.shape
          face_3d_landmarks = []
          for idx, lm in enumerate(face_landmarks.landmark):
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            depth_mm = depth_frame_resized[y_px, x_px] if 0 <= x_px < w and 0 <= y_px < h else 0
            face_3d_landmarks.append((x_px, y_px, depth_mm))
            # Draw label if this is a key landmark
            if idx in landmark_labels:
              label = f"{landmark_labels[idx]}"
              cv2.putText(frame_out, label, (x_px, y_px-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
              cv2.putText(frame_out, f"Z:{depth_mm}mm", (x_px, y_px+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
          # Example: print first landmark's 3D coordinates
          if face_3d_landmarks:
            print("First landmark 3D:", face_3d_landmarks[0])

      cv2.imshow('DepthAI Face Mesh 3D', cv2.flip(frame_out, 1))
      if cv2.waitKey(1) & 0xFF == 27:
        break
  cv2.destroyAllWindows()