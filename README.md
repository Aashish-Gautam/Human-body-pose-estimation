# Human-body-pose-estimation
To categorize human body pose into standing, sitting and other using computer vision from any image

Estimation model used - Mediapipe

Relevant landmarks for pose detection (among Sitting, Standing and Unknown) -

1. Left shoulder
2. Right shoulder
3. Left hip
4. Right hip
5. Left knee
6. Right knee
7. Left ankle
8. Right ankle

Relevant new coordinates from the obtained landmarks -

1. Mid-chest coordinates = average of left and right shoulder coordinates
2. Hip coordinates = average of left and right hip coordinates

Relevant vectors from obtained landmarks and coordinates -

1. upper_body_vector = hip_coordinates - midchest_coordinates
2. left_thigh_vector = left_hip_coordinates - left_knee_coordinates
3. right_thigh_vector = right_hip_coordinates - right_knee_coordinates
4. left_calves_vector = left_ankle_coordinates - left_knee_coordinates
5. right_calves_vector = right_ankle_coordinates - right_knee_coordinates

Relevant angles calculated using above obtained vectors -

1. left_bending_angle = angle formed by upper_body_vector and left_thigh_vector
2. right_bending_angle = angle formed by upper_body_vector and right_thigh_vector

Relevant inclinations with the horizontal of the obtained vectors -

1. upper_body_inclination : inclination of upper_body_vector with the horizontal
2. left_calves_inclination : inclination of left_calves_vector with the horizontal
3. right_calves_inclination : inclination of right_calves_vector with the horizontal

## Conditions for Standing pose - 

1. Upper body inclination is greater than 60 degrees from horizontal
2. Left and right calves inclinations are greater than 60 degrees with horizontal
3. Left and right bending angles are greater than 150 degrees

## Conditions for Sitting pose - 

1. Upper body inclination is less than 30 degrees from horizontal
2. Left and right bending angles are less than 150 degrees

If the conditions of Standing and Sitting are not satisfied then the pose is “Unknown”.

# Output Examples:

1. Standing
![proc_a](https://github.com/Aashish-Gautam/Human-body-pose-estimation/assets/54966966/b107bcae-cfc0-42b4-8b46-07ee35be627b)

2. Sitting
![proc_e](https://github.com/Aashish-Gautam/Human-body-pose-estimation/assets/54966966/daa79bf1-fe9d-4c03-a2a4-6a313b0bc652)
