# Lego-Mindstorms-Terrain-Classifier
This is a terrain classifier that looks at using a custom built Lego MindStorm 51515 robot to collect data and classify terrain.


Overview
This project utilises a LEGO Mindstorms 51515 robot to classify different types of terrain based on sensor data collected during navigation.
The project consists of several Python files, each designed for specific tasks including data collection, data visualisation, feature extraction,
model training, evaluation, and experiments to assess the impact of feature reduction and classification time reduction.


Code and Usage
1. Terrain_classifier_data_collection.py
Description: This code is responsible for collecting data from the robot's sensors as it moves over different terrains. 
It captures a variety of measurements such as motor encoder positions, IMU tilt angles, accelerations, angular velocities, and ultrasonic sensor readings.
Usage: Run this code while the robot navigates over the terrain to gather the necessary data for terrain classification.

2. Visualise_Collected_Data.py
Description: Utilises matplotlib to visualise the data collected from the robot. This code helps in understanding the sensor outputs
and in verifying the integrity of the data collection process.
Usage: Execute this code after collecting data to generate plots of the different sensor readings.
Ensure the data path in the script points to the collected data files.

3. Terrain_classifier_FICV.py
Description: Stands for Feature Importance and Cross-Validation. This code loads the sensor data, extracts features,
and applies various machine learning models to classify the terrain. It evaluates model performance using cross-validation and generates 
confusion matrices and classification reports.
Usage: This code is used for training the classification models.
There are additional files with the name Terrain_classifer but have a combination of FICV, each file is an experiment to see
how the classifer performs with/without cross validation and feature importance.

4. Terrain_Classifier_Feature_Importance_Experiment.py
Description: Code for an experiment that investigates the impact of each sensor feature on the classification accuracy. It performs feature importance
analysis and evaluates models with reduced feature sets.
Usage: Run to see how model performance changes with the removal of certain features. Useful for optimising the model to improve computational efficiency or accuracy.

5. Time_Till_Classify.py
Description: Code for an experiment that tests how quickly the robot can classify terrain by reducing the amount of data needed for a classification decision.
It's useful for scenarios where rapid terrain classification is critical.
Usage: Useful for performance testing in different operational scenarios. Adjust rows_per_set to change the amount of data used for classification.


How it Works Together
Data Collection: Start with Terrain_classifier_data_collection.py to collect terrain data as the robot navigates different surfaces.
Visualisation: Use Visualise_Collected_Data.py to check and visualise the data collected, ensuring the sensors are functioning as expected.
Model Training and Evaluation: With Terrain_classifier_FICV.py, train various models using the visualised data and evaluate their performance.
Feature Importance Analysis: Terrain_Classifier_Feature_Importance_Experiment.py helps in understanding which features are most impactful,
allowing for refinement of the data collection and feature extraction processes.
Classification Speed Testing: Finally, Time_Till_Classify.py assesses the efficiency of the model by determining the minimum amount of data required for accurate classification.

Getting Started
To get started with this project, ensure you have the necessary Python environment and libraries installed, including Pybricks, pandas, numpy, scikit-learn, lightgbm, and matplotlib. Follow the usage sections to see how to run these bits of code

