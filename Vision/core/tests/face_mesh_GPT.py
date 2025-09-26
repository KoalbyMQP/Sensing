"""
Generated with the help of GPT-4.1 from face_mesh.py and obj_tracker.py
"""
# Dependencies (see install requirements and README)
import cv2 # from OpenCV Python lib
import depthai as dai # Luxonis DepthAI v3.0.0 -- TODO: migrate to Depthai ROS when possible
import mediapipe as mp # MediaPipe v0.10.21
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Instantiate pipeline from DepthAI
pipeline = dai.Pipeline()

# Setup RGB camera interface
cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
rgb_output = cam_rgb.requestOutput((640, 480))
rgb_queue = rgb_output.createOutputQueue()

# Setup infrared cameras interface
mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
left_output = mono_left.requestOutput((640, 400))
right_output = mono_right.requestOutput((640, 400))
left_queue = left_output.createOutputQueue()
right_queue = right_output.createOutputQueue()
# Calculate depth from IR camera inputs
stereo = pipeline.create(dai.node.StereoDepth)
left_output.link(stereo.left)
right_output.link(stereo.right)
depth_output = stereo.depth # FIXME: derive depth calculation from SpatialDetectionNetwork and ObjectTracker device nodes
depth_queue = depth_output.createOutputQueue()

# Start pipeline
with pipeline:
  pipeline.start()
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  with mp_face_mesh.FaceMesh(
    max_num_faces=10, # can only recognize up to 10 human faces at a time
    refine_landmarks=True,
    min_detection_confidence=0.5, # may lead to false positives (artifacts) if max_num_faces > 1
    min_tracking_confidence=0.5) as face_mesh:
    while pipeline.isRunning():
      in_rgb = rgb_queue.get()
      in_depth = depth_queue.get()
      frame = cv2.flip(in_rgb.getCvFrame(), 1)
      depth_frame = in_depth.getFrame() # 16-bit depth map
      # Resize depth frame to match RGB frame size
      depth_frame_resized = cv2.resize(depth_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

      # RGB input to MediaPipe
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(rgb_frame)

      # Create face mesh
      frame_out = frame.copy()
      if results.multi_face_landmarks:
        # Key landmark indices and labels
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

          # Get coordinates of landmarks
          h, w, _ = frame.shape
          face_3d_landmarks = []
          for idx, lm in enumerate(face_landmarks.landmark):
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            depth_mm = depth_frame_resized[y_px, x_px] if 0 <= x_px < w and 0 <= y_px < h else 0
            face_3d_landmarks.append((x_px, y_px, depth_mm))
            # Draw label if key landmark
            if idx in landmark_labels:
              label = f"{landmark_labels[idx]}"
              # cv2.putText(frame_out, label, (x_px, y_px-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2) # eye labels are incorrect due to frame flip
              cv2.putText(frame_out, f"x: {x_px} mm", (x_px, y_px-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
              cv2.putText(frame_out, f"y: {y_px} mm", (x_px, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
              # cv2.putText(frame_out, f"z: {depth_mm} mm", (x_px, y_px+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1) # depth_frame is often a singular matrix populated primarily with zeroes

      cv2.imshow('OAK-D Lite Face Mesh', frame_out)
      if cv2.waitKey(1) & 0xFF in (ord('q'),27): # press 'q' or 'ESC' to exit window
        break
  cv2.destroyAllWindows()