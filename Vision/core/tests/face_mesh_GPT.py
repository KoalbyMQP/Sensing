# https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html
# https://docs.luxonis.com/software-v3/depthai/examples/image_align/depth_align
# https://docs.luxonis.com/software-v3/depthai/examples/spatial_location_calculator/spatial_location_calculator/
"""
Scaffolding generated with minimal input from GPT-4.1 from face_mesh.py and obj_tracker.py (see `conversation.md`)
"""
import cv2 # from OpenCV Python lib
import depthai as dai # Luxonis DepthAI v3.0.0 -- TODO: migrate to Depthai ROS when possible
import mediapipe as mp # MediaPipe v0.10.21
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

colour = (255, 255, 255)

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
rgb_in = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A) # RGB camera
left_in = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B) # left IR camera
right_in = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C) # right IR camera

# Nodes
stereo = pipeline.create(dai.node.StereoDepth) # for linking depth
spatial = pipeline.create(dai.node.SpatialLocationCalculator) # for spatial calculations
sync = pipeline.create(dai.node.Sync) # for colour camera
# Inputs
rgb_out = rgb_in.requestOutput((1280, 720), enableUndistortion=True)
left_out = left_in.requestOutput((640, 480))
right_out = right_in.requestOutput((640, 480))
# Linking
rgb_out.link(sync.inputs['RGB'])
left_out.link(stereo.left)
right_out.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)

config = dai.SpatialLocationCalculatorConfigData()
config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MODE
config.depthThresholds.lowerThreshold = 10
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(0,0,0,0) # define region of interest (ROI) for spatio-depth calculations

spatial.inputConfig.setWaitForMessage(False)
spatial.initialConfig.addROI(config)

spatial_queue = spatial.out.createOutputQueue()
depth_queue = spatial.passthroughDepth.createOutputQueue()
rgb_queue = sync.out.createOutputQueue()

stereo.depth.link(spatial.inputDepth)
rgb_out.link(stereo.inputAlignTo)

inputConfigQueue = spatial.inputConfig.createInputQueue()

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)

mouse_coords=[0,0];old_mouse_coords=mouse_coords[:]
def mouse_callback(event,x,y,flags,param):mouse_coords[0]=x;mouse_coords[1]=y
windowName = "OAK-D Lite Face Mesh"
depthWeight = 0
colourWeight = 1
show_face_mesh = True
filepath = 'face_mesh_GPT.png'

with pipeline:
    pipeline.start()
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 1280, 720)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        max_num_faces=10, # can only recognize up to 10 human faces at a time
        refine_landmarks=True,
        min_detection_confidence=0.5, # may lead to false positives (artifacts) if max_num_faces > 1
        min_tracking_confidence=0.5
    ) as face_mesh:
        while pipeline.isRunning():
            spatial_data = spatial_queue.get().getSpatialLocations()

            rgb_data = rgb_queue.get()
            rgb_frame = rgb_data['RGB']

            outputDepthImage : dai.ImgFrame = depth_queue.get()

            frameDepth = outputDepthImage.getFrame() # getCvFrame is technically slower but handles overhead
            cvFrame = rgb_frame.getCvFrame()

            depthFrameColor = cv2.normalize(frameDepth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(-depthFrameColor, cv2.COLORMAP_INFERNO)

            blended = cv2.addWeighted(
                cvFrame, colourWeight, depthFrameColor, depthWeight, 0
            )
            h, w, _ = cvFrame.shape

            if len(spatial_data):
                depth_data_cursor = spatial_data[0] # first ROI is guaranteed to be that of the cursor
                roi = depth_data_cursor.config.roi
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)

                fontType = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(blended, (xmin, ymin), (xmax, ymax), colour, 1)
                cv2.putText(blended, f"x: {int(depth_data_cursor.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.3, colour, 1)
                cv2.putText(blended, f"y: {int(depth_data_cursor.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.3, colour, 1)
                cv2.putText(blended, f"z: {int(depth_data_cursor.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.3, colour, 1)

            results = face_mesh.process(cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB))
            # frame_out = blended.copy() # using `blended` from this point onwards may be reckless
            if (results.multi_face_landmarks!=None) & show_face_mesh:
                # Key landmark indices and labels
                landmark_labels = {
                1: "Nose Tip",
                33: "Left Eye",
                263: "Right Eye",
                61: "Mouth",
                151: "Forehead"
                }
                forehead_labels = {
                    i:"Forehead" for i in [162,71,63,105,66,107,109,67,103,54,21,108,69,104,68,389,301,293,334,296,336,9,151,10,338,297,332,284,251,337,299,333,298]
                }
                configs = [config]
                landmarks_list = []
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=blended,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=blended,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=blended,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                    # Get coordinates of landmarks
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in landmark_labels:
                            config_lm = dai.SpatialLocationCalculatorConfigData()
                            config_lm.depthThresholds.lowerThreshold = config.depthThresholds.lowerThreshold
                            config_lm.depthThresholds.upperThreshold = config.depthThresholds.upperThreshold
                            config_lm.roi = dai.Rect(int(w*lm.x),int(h*lm.y),2,2)
                            configs.append(config_lm)
                            landmarks_list.append([idx, lm])
                        # Draw forehead dots
                        if idx in forehead_labels:
                            cv2.circle(blended, (int(w*lm.x), int(h*lm.y)), 2, (255, 0, 0), -1)
                
                if len(configs):
                    cfg = dai.SpatialLocationCalculatorConfig()
                    cfg.setROIs(configs)
                    inputConfigQueue.send(cfg)
                j=0
                for depth_data in spatial_data:
                    if j:
                        x_px = round(w*landmarks_list[j-1][1].x)
                        y_px = round(h*landmarks_list[j-1][1].y)
                        cv2.putText(blended, f"{landmark_labels[landmarks_list[j-1][0]]}", (x_px, y_px-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                        cv2.putText(blended, f"x: {int(depth_data.spatialCoordinates.x)} mm", (x_px, y_px-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                        cv2.putText(blended, f"y: {int(depth_data.spatialCoordinates.y)} mm", (x_px, y_px+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                        cv2.putText(blended, f"z: {int(depth_data.spatialCoordinates.z)} mm", (x_px, y_px+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                    j+=1

            # Show the frame
            cv2.imshow(windowName, blended)
            cv2.setMouseCallback(windowName, mouse_callback)
            if old_mouse_coords != mouse_coords:
                old_mouse_coords = mouse_coords[:]
                config.roi = dai.Rect(mouse_coords[0],mouse_coords[1],2,2)
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                inputConfigQueue.send(cfg)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('1'): # press '1' to switch to depth view
                depthWeight=1;colourWeight=0
            elif key & 0xFF == ord('2'): # press '2' to switch to combined view
                depthWeight=0.5;colourWeight=0.5
            elif key & 0xFF == ord('3'): # press '3' to switch to colour view
                depthWeight=0;colourWeight=1
            elif key & 0xFF == ord('m'): # press 'M' to toggle face mesh
                show_face_mesh = False if show_face_mesh else True
            elif key & 0xFF == ord('s'): # press 'S' to save screenshot
                cv2.imwrite(filepath, blended)
            elif key & 0xFF in (ord('q'),27): # press 'Q' or 'ESC' to exit window
                break