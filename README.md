### Decoding-the-Mind
Mapping Visual Perception with EEG and Deep Learning
- Understanding how the human brain processes visual stimuli is a key challenge in neuroscience. In this project, we explore the relationship between EEG data and visual stimuli classification using deep learning models. By leveraging feature extraction from AlexNet, ResNet and MOCO and dimensionality reduction via PCA, we aim to bridge the gap between neural activity and machine learning predictions.

# Our approach focuses on:
- Implementing feature extraction on our training dataset of 16540 images.
- Utilizing PCA to reduce dimensionality & optimize our computational efficiency.
- Performing linear regression to fit PCA on to our extracted features.

# Motivation: 
- The Motivation behind our project was the Things Initiative, specifically Things EEG 2. We wanted to model human vision recognition using various DNNs in order to accurately classify the EEG signal a person give off to the image they were looking at when the signal was captures. We also wanted to compare the performance of AlexNet, MOCO, and ResNet 18 on feature extractions and their ability to accurately bridge those feature maps to the correct EEG signal. 
- Work like this can help us to understand how our brain comprehends visual information in a way that can be applied to computer vision and its object recognition abilities. This can lead to improvements in feature extraction methods of EEGs which can improve methods of diagnosing as many diseases such as Alzheimer's can be linked to abnormal EEG signals. Ultimately, as we continue to model computers off of the brain, and conversely, understand the brain through the lens of computers, we being to shed light on the unknowns of our cognitions. 


# Dataset:
EEG Signals: Preprocessed neural activity data from 16,540 trials across 17 channels.
Image Data: Images were passed through a pre-trained DNN to extract feature maps.
            PCA was applied to reduce dimensionality of the feature maps, resulting in components per trial.

# Data Preprocessing:
- EEG signals were flattened into a matrix of shape (16500, channels × time points).
- EEG data was normalized using StandardScaler.

# DNN Preload:
- Identifying Layers: Early EEG responses are better correlated with features from shallow layers, which capture basic visual information processed in the brain shortly after stimulus onset.
- Extracting Feature Maps: Extracting and appending feature maps from layers allows us to create a compact, high-dimensional representation of the image features.
- Importance: By selecting specific layers, we could investigate how well each stage of the DNN aligns with corresponding stages of visual processing in the brain, providing insights into brain-computer parallels. This mapping of layers to EEG time points mirrors how the brain processes visual information hierarchically over time. The combination of features from multiple layers ensures the models capture both low-level (e.g., edges) and high-level (e.g., objects) visual information, leading to better predictions of the EEG data across time points.
- PCA across Feature Maps

# PCA across Feature Maps:
- Dimensionality Reduction: The DNN feature maps generated from the layers contain very high-dimensional data. PCA was used to reduce this dimensionality while retaining the most informative features, optimizing the computational efficiency of the encoding models. 
- Feature Selection: The study reduced the feature maps to 100/50 principal components, ensuring that the most significant patterns were preserved. This selection was necessary to focus on components that explained the most variance in the data, which aligns with the goal of modeling neural responses effectively.

# EEG Linear Regression:
- Linear Regression Setup:EEG signals (predictors) were mapped to image PCA components (targets).
Model trained to learn linear relationships between EEG features and PCA components.
Performed analysis on all trials using the scikit-learn LinearRegression implementation.
- Evaluation Metrics:
R-squared Score: Quantifies the proportion of variance in PCA components explained by EEG signals.



# Key Results (Moco): R-squared Score: 0.4386:
- Approximately 43.86% of the variance in PCA components derived from DNN feature maps is explained by EEG signals.
- Scatter plot for first PCA component shows a moderate positive correlation.
- For sake of computational expense, the figure shows only the linear regression of PCA component 1 of the actual image and its predicted EEG signal
- The regression shows a strong trend of accurate prediction from image to the EEG signal a person gives off when viewing the image
- While not entirely linear, the plot shows that the PCAs of images with variances varying from -60 to 60 line up enough to display a positive correlation between the two

# What Worked:
- Moderate Predictive Power: Linear regression achieved an R-squared score of 0.4386, explaining ~43.86% of the variance in image PCA components using EEG signals.
- First PCA Component (Moco): Scatter plot showed a positive linear trend between predicted and actual values, confirming alignment for PCA Component 
- Scalable Pipeline (Moco): Successfully handled large EEG and image datasets, incorporating preprocessing, PCA, and regression without major bottlenecks.

# What didn’t:
ResNet: We got a working linear regression, however, it had an R-squared score of 0.0003 so the regression model was ineffective at explaining the variance in image PCA components. 
AlexNet: Due to AlexNet being fairly outdated compared to modern DNNs, although we were able to easily generate the feature maps of all of the images, the quality of those feature maps were not sufficient for our classification goal of EEG to image. 

# Learnings:
- There is a discernible linear relationship between neural activity and high-level visual features extracted by deep neural networks (DNNs), though it is far from perfect.
- Pre-trained DNNs, such as AlexNet and ResNet, provide robust feature representations, making them suitable for studies bridging neural and computer vision domains.
- EEG signals contain meaningful information about image feature maps, as demonstrated by the moderate R-squared score (0.4386) from linear regression.

