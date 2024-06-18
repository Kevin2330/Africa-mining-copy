# Mining Mines: A Case Study on Detecting Mining Activities in Eastern Congo Using Satellite Imagery and Scalable Computing Methods

## Introduction

The Eastern Congo is a region rich in minerals such as gold, tantalum, and tin, which are vital for various industries worldwide. However, the artisanal mining activities in this area are often unregulated, leading to significant environmental degradation and severe social issues, including human rights abuses and armed conflicts. Monitoring and regulating these activities is crucial for mitigating their negative impacts. Traditional methods of monitoring, such as on-ground inspections, are often impractical due to the region's remote and conflict-prone nature. Therefore, leveraging advanced technologies like satellite imagery and scalable computing methods offers a promising solution to this challenge.

Our project focuses on detecting artisan mining activities in Eastern Congo using satellite imagery data from the Landsat dataset hosted on AWS. The primary research problem is to develop an accurate and efficient method for identifying active mining areas using remote sensing techniques. This task involves processing vast amounts of satellite data, which requires substantial computational resources. By employing scalable computing methods, we aim to create a model that can be used for continuous monitoring and regulation of mining activities, ultimately contributing to the region's sustainable development.


## Justification for Using Scalable Computing Methods

1. Volume of Data
The Landsat dataset comprises decades of satellite imagery, capturing vast geographical areas with high temporal frequency. Analyzing this data manually is not feasible due to its sheer volume. Scalable computing methods, such as cloud computing and distributed processing, enable us to handle and analyze large datasets efficiently.

2. High-Resolution Analysis
Detecting mining activities requires high-resolution analysis to distinguish between different land use types accurately. This precision demands significant computational power, which can be achieved using scalable methods like GPU computing and parallel processing.

3. Model Training and Evaluation
Developing a robust machine learning model necessitates training on large datasets to ensure accuracy and generalizability. This process involves numerous iterations and fine-tuning, which are computationally intensive. Scalable computing allows for faster training and evaluation, reducing the overall project timeline.

4. Real-Time Monitoring
For the model to be effective in real-world applications, it must be capable of real-time monitoring. This requires processing and analyzing new satellite images as they become available. Scalable computing infrastructures can support continuous data ingestion and real-time analysis, making timely interventions possible.

## Methodology


- **Data Collection**: Utilize AWS Lambda functions and step function for data collection(read satellite tiffs file using rasterio, calculate NDVI(Normalized Difference Vegetation Index) loss, filter locations with no cloud cover and ndvi loss bigger than 0.15) [1] and store the collected data and band feature in csv format an AWS S3 bucket.
- **Data Processing**: Process and append the feature data acquired from sources other than landsat dataset using the Midway cluster (append few feature to orginal dataset). MPI (Message Passing Interface) is used for parallel processing of mining location data to efficiently calculate distances to various geographic features.
- **Machine Learning**: Create an Amazon EMR cluster to perform data cleaning, processing, model training, and visualization using AWS PySpark sessions.

![image](https://github.com/macs30123-s24/final-project-mining-mines/assets/143442308/e7e8e5d0-5f6b-4d44-a078-b29fca80b071)

## Architecture diagram

![Architecture Diagram](https://github.com/macs30123-s24/final-project-mining-mines/blob/main/architecture.png)

## 1. Data Collection and Preprocessing

### 1.1 Data Source

Data Source: We have three sources of data, the first is true(existing) artisanal mine locations from [IPIS Geoserver Map Preview](https://geo.ipisresearch.be/geoserver/web/wicket/bookmarkable/org.geoserver.web.demo.MapPreviewPage?0). The second is landsat data to generate ndvi loss and variaous band features of a location, the third is human activity data (location of village, natural protected area, road and water way) [Humanitarian Data Exchange](https://data.humdata.org/dataset/central-african-republic-roads) to generate other features of a location. 

### 1.2 Collect Landsat Data using Lamda Function and Step Function: 
We use AWS "gsas-landsat" S3 buckets to retrive filtered landsat data of east congo: we used [lambda function](https://github.com/macs30123-s24/final-project-mining-mines/blob/main/lambda_function.py) to filter no cloud cover data to calculate NDVI loss and filter locations with ndvi loss > 0.15 [1], and we also retrive data of different band lenth features of each location. After obtaining the filtered data with band features, the focus was adding other features of distances and matching data with true artisanal mine locations(step 1.3 and step 1.4)

We reference a paper from Science of The Total Environment and get the idea that high NDVI（ Normalized Difference Vegetation Index） loss indicates a low level of land cover and a high level of human activity, including mining activity. This means that to improve accuracy and reduce data amount, we can get the training database by filtering areas with high NDVI loss. Then, we classify these high human activity areas into mining and non-mining locations. We first begin with calculating NDVI loss; we have two groups of data, reference period data (2003-2005 landsat 8) and observed period data(2020-2023 landsat8). We calculate loss based on the difference in NDVI value between the two periods. Then, we filter out locations with NDVI loss greater than 0.15.


- to deploy lambda function and step function, please see [step1.ipynb](https://github.com/macs30123-s24/final-project-mining-mines/blob/main/step1-get-data.ipynb)

(Initially, we attempted to use Google Earth Engine to select areas in a specific part of the DRC with NDVI loss > 0.15 [1]. However, this process took a long time without the possibility of parallelization and did not yield many data points (around 4000 points in an area of 4 latitude by 3 longitude)

### 1.3 Add Other Human Activity Features
We initially want to append other human activity features using pyspark, but due to error with dependencies, we failed to do that intially. Thus we use midway cluster to do add features.
We calculated the distance to the nearest road or features(Nearest Waterways, Local Roads, Main Roads, Protected Areas, and Villages) using GeoPandas and GIS analysis.

- to add other features, please see [code](https://github.com/macs30123-s24/final-project-mining-mines/blob/main/Data-preprocessing/Data-Preprocessing-Midway/add_feature_5k_y1.py) This script calculates the distance to various features for NDVI loss points within a 5000-meter radius of mining locations.

### 1.4 Labelling Strategy
We need to label artisanal mines with ndvi loss greater than 0.15 [1] retrived from the previous step into [true mine](https://github.com/macs30123-s24/final-project-mining-mines/blob/main/Data-preprocessing/Complete-Data/complete_data_y1.csv) and [not mines](https://github.com/macs30123-s24/final-project-mining-mines/blob/main/Data-preprocessing/Complete-Data/complete_data_y0_200k.csv), using data of [true mine location data](https://geo.ipisresearch.be/geoserver/web/wicket/bookmarkable/org.geoserver.web.demo.MapPreviewPage?0=). (As a result, we added a additional column to data we got from 1.3)

**Labelling**: 
   - **Strategy**: Use the mining location as the center of a circle, select a radius, and see which locations in the initial CSV file are contained within the circles.
   - **Labeling**: Points contained within the circle are labeled as 1, and others as 0.
   - **Radius Used**: 1000 meters and 5000 meters.
   - **Implementation**: Initially attempted on PySpark, but due to environment configuration issues, the process was run on the Midway SSD cluster
-  **to match with existing mine**: [match_with_existing_mine.py](https://github.com/macs30123-s24/final-project-mining-mines/blob/main/Data-preprocessing/Data-Preprocessing-Midway/match_with_existing_mine.py)
  - **Description**: This script matches NDVI loss points with existing mine locations and labels them based on their proximity to the mines.

**Justification for Choosing 1000 Meters and 5000 Meters**:

- **1000 meters**: This smaller radius helps in accurately identifying points that are very close to the mine, ensuring that the labeling is precise for areas immediately around the mine. This can help in understanding the direct impact zone of mining activities.
  
- **5000 meters**: This larger radius allows us to capture the broader impact zone of mining activities. It takes into account potential indirect effects such as transportation routes, dust dispersion, and other environmental impacts that might extend beyond the immediate vicinity of the mine.

Using these two radii helps in creating a detailed and comprehensive analysis of the impact zones, providing insights into both immediate and extended effects of mining activities on NDVI loss.

### 1.5 Output

The results of features and label of each location of east congo are saved to a new CSV file to s3 bucket(bucketname = "africamining"), which includes the potential mining indicator for each NDVI loss point.

## 2. Machine Learning
[machine learning and pipeline code](https://github.com/macs30123-s24/final-project-mining-mines/blob/main/ml_pipelines.ipynb)

After buliding EMR cluster, we use two sample datasets (radius 1km vs 5km) and  logit model, random forest, and neutral network models to train and predict data on Spark Session. We have three main steps: 
- Step 1: Data Processing and Visualization
- Step 2: Data Transformations and Feature Engineering 
- Step 3: Develop Pipelines and Employee Machine Learnig Models to Predict.

Below are confusion matrix of above  machine learning models:

![confusion matrix_1km](https://github.com/macs30123-s24/final-project-mining-mines/blob/b77f04ffa2557fd420f8a54a201289c1de774b62/41716684776_.pic.jpg)

![confusion matrix_5km](https://github.com/macs30123-s24/final-project-mining-mines/blob/b77f04ffa2557fd420f8a54a201289c1de774b62/51716684777_.pic.jpg)

  
## 3. Conclusion

### Sample Data Using Radius 1km as Cutoff

| Model               | Logit Model           | Random Forest        | Neural Network        |
|---------------------|-----------------------|-----------------------|-----------------------|
| **Test Accuracy**   | 0.817176979           | 0.873840681           | 0.823840681           |
| **True Positive Rate** | 0.802498048        | 0.837837838           | 0.756192661           |
| **True Negative Rate** | 0.831181728        | 0.91541199            | 0.844488875           |


### Sample Data Using Radius 5km as Cutoff

| Model               | Logit Model           | Random Forests        | Neural Network        |
|---------------------|-----------------------|-----------------------|-----------------------|
| **Test Accuracy**   | 0.773996431           | 0.833940681           | 0.783840681           |
| **True Positive Rate** | 0.796872528        | 0.7663320331649509    | 0.747660243           |
| **True Negative Rate** | 0.752295184        | 0.9364751452550032    | 0.873700306           |


- **Best Performing Model**: The Random Forests model outperforms the Logit Model and Neural Network in both accuracy and the ability to correctly identify mining and non-mining activities (highest true positive and true negative rates). This makes it the most reliable model for detecting potential mining activities.
- **Impact of Radius Cutoff**: Increasing the radius cutoff from 1km to 5km generally decreases the performance of the Logit Model and Neural Network, while the Random Forests model maintains consistent performance. This suggests that the Random Forests model is more robust to changes in spatial scale.
- **Model Recommendations**: For applications requiring high accuracy and reliability in detecting mining activities, the Random Forests model is recommended. The Logit Model, while simpler, performs adequately but not as well as the Random Forests model. The Neural Network shows potential but may require further tuning to handle larger spatial scales effectively.

In summary, for both radius cutoffs, the Random Forests model demonstrates superior performance, making it the preferred choice for detecting potential mining activities in the given dataset.




## 4. Appendix: Decriptive analysis of data

The analysis considers five key geographic features: waterways, protected areas, main roads, local roads, and village distributions. The following visualizations represent these features:

- the code to generate descirptive visualization:
- **Congo Visualization**: [Congo-visualization.ipynb](https://github.com/macs30123-s24/final-project-mining-mines/blob/main/Data-preprocessing/Notebooks/Congo-visualization.ipynb).

### Example Visualizations

1. **Waterways and Specific Point**
   ![image](https://github.com/Huiyu1999/African-mining/assets/143442308/226e69b8-1a58-480e-aa24-6b4da9120b3e)

   - **Description**: This map shows the distribution of waterways (in blue) in the DRC, with a specific point highlighted in red. It helps in analyzing the proximity of mining locations to water sources.

2. **Protected Areas and Specific Point**
   ![image](https://github.com/Huiyu1999/African-mining/assets/143442308/0b9a7219-57f0-416f-bda8-82cf2bb26819)

   - **Description**: This map displays the protected areas (in green) in the DRC, with a specific point marked in red. It is used to assess the potential impact of mining activities on protected regions.


## Team Roles 

**Yu Hui**: Processing landsat data as well as related feature using AWS 

**Kevin**:  Data pre-processinng and gnenerate related features.

**Charlotte**: Responsible mainly for training the machine learning model, analyze the performance, and conducting respective visualization.

## Reference

[1] Hilson, G. (2021). The large footprint of small-scale artisanal gold mining in Ghana. Science of The Total Environment, 778, 146331. https://doi.org/10.1016/j.scitotenv.2021.146331



