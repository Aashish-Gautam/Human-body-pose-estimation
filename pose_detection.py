import cv2
import mediapipe as mp
import numpy as np

def detect_pose(input_image_path,output_image_path):
  #read image
  image = cv2.imread(input_image_path)

  # Load the MediaPipe Pose model
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

  # Convert the image to RGB
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Process the image with MediaPipe Pose
  results = pose.process(image_rgb)
  pose_landmarks=results.pose_world_landmarks

  #Relevant location coordinates and angles needs to detect posture of the person in the image
  left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
  left_shoulder_coords = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])

  left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
  left_hip_coords = np.array([left_hip.x, left_hip.y, left_hip.z])

  left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
  left_knee_coords = np.array([left_knee.x, left_knee.y, left_knee.z])

  left_upper_body_vector = left_hip_coords - left_shoulder_coords
  left_thigh_vector = left_hip_coords - left_knee_coords

  right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
  right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
  right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

  right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
  right_shoulder_coords = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])

  right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
  right_hip_coords = np.array([right_hip.x, right_hip.y, right_hip.z])

  right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
  right_knee_coords = np.array([right_knee.x, right_knee.y, right_knee.z])

  right_upper_body_vector = right_hip_coords - right_shoulder_coords
  right_thigh_vector = right_hip_coords - right_knee_coords

  hip_coords=(left_hip_coords+right_hip_coords)/2.0
  midchest_coords=(left_shoulder_coords+right_shoulder_coords)/2.0

  upper_body_vector=hip_coords - midchest_coords

  left_bend_angle=np.arccos(np.dot(upper_body_vector,left_thigh_vector)/(np.linalg.norm(upper_body_vector)*np.linalg.norm(left_thigh_vector)))
  right_bend_angle = np.arccos(np.dot(upper_body_vector, right_thigh_vector) / (np.linalg.norm(upper_body_vector) * np.linalg.norm(right_thigh_vector)))

  left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
  left_ankle_coords = np.array([left_ankle.x, left_ankle.y, left_ankle.z])

  right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
  right_ankle_coords = np.array([right_ankle.x, right_ankle.y, right_ankle.z])

  left_calves_vector = left_ankle_coords - left_knee_coords
  right_calves_vector = right_ankle_coords - right_knee_coords

  # Function to detect inclination of a vector w.r.t horizontal
  def inclination(vector):
    x=vector[0]
    y=vector[1]
    z=vector[2]
    return np.arctan(y/((x**2+z**2)**0.5))

  # Inclination of relevant body parts
  upper_body_inclination=inclination(upper_body_vector)
  left_calves_inclination=inclination(left_calves_vector)
  right_calves_inclination=inclination(right_calves_vector)

  # Defining a pose (Standing,sitting or unkown) from obtained angles and inclinations
  pose="Unknown"
  # for standing
  if upper_body_inclination >np.radians(60) and left_calves_inclination>np.radians(60) and right_calves_inclination>np.radians(60) and left_bend_angle>np.radians(150) and right_bend_angle>np.radians(150):
    pose="Standing"

  # for sitting
  elif upper_body_inclination >np.radians(30) and left_bend_angle<np.radians(150) and right_bend_angle<np.radians(150):
    pose="Sitting"

  # Write the pose on the image
  img=cv2.imread(input_image_path)

  text = pose
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  font_thickness = 1
  text_color = (255, 255, 255)  # White
  background_color = (0, 0, 0)  # Black

  text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
  text_width, text_height = text_size

  x = (img.shape[1] - text_width) // 2
  y = (img.shape[0] + text_height) // 2
  start_point,end_point = (x, y - text_height), (x + text_width, y)

  cv2.rectangle(img,start_point,end_point, background_color, -1)
  cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

  img_name=(input_image_path.split('/'))[-1]
  cv2.imwrite(output_image_path+'proc_'+img_name, img)
