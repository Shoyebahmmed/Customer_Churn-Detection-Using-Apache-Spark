# Project Title: Customer Churn Detection Using Apache Spark

## 1. Overview
For many organizations, particularly those that depend on subscriptions or repeat revenue, losing customers is a common problem. Businesses lose money when customers decide to leave, so being able to anticipate churn early on may help organizations increase customer satisfaction and take appropriate action.

The goal of this project is to combine MLlib and Apache Spark to create a prototype for customer churn prediction. The aim is to demonstrate how a machine learning model that predicts which customers are likely to churn can be trained using large-scale consumer data processing.

This repository includes:

- A working churn-prediction pipeline built with Apache Spark
- Steps for data loading, cleaning, and feature preparation
- Model training and evaluation using MLlib
- A batch-style processing workflow

Rather than a complete production solution, the prototype illustrates the main concept and technological procedure. It demonstrates how Spark may aid in the analysis of huge datasets and facilitate data-driven choices to lower customer attrition.

## 2. Project Objectives
The primary objective of this project is to develop a functional prototype that demonstrates how large-scale data processing can be used to predict client loss. More specifically, the project aims to:

- Showcase Spark's ability to manage and process data for machine learning in an effective manner.
- Choose and modify the factors that affect customer attrition to prepare the dataset.
- Compare several models (such as Logistic Regression and Random Forest) and assess performance using measures like AUC with MLlib.
- Create a procedure that can be executed in batches so that forecasts can be easily updated whenever new data is received.

## 3. Repository Structure
This repository is structured such that documentation, sample data, and prototype code are all properly separated. The churn prediction pipeline's layout makes it simple to comprehend how each component functions.

```text
project-root/
├── churn_data.csv
├── churn_prediction.ipynb
├── architecture.png
├── README.md
```

## 4. System Architecture 
### 4.1 Current Implementation (Prototype)
The goal of this prototype is to combine PySpark and a structured dataset to create a churn prediction model. It operates as a batch process in which machine learning models are trained and tested using the entire dataset at once.

**Key steps in the prototype architecture:**

#### Data Ingestion
- The system reads a CSV dataset directly into a Spark DataFrame.
- All records are processed together in a single batch.

#### Data Preparation
- Features such as Age, Total_Purchase, Years, and Num_Sites are selected and transformed.
- VectorAssembler is used to combine selected columns into a feature vector.

#### Model Training & Evaluation
- Three machine learning models are trained using Spark MLlib:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosted Trees
- Models are evaluated using AUC (Area Under ROC) to measure prediction performance.

#### Batch-Based Execution
- Since the data is static and loaded all at once, this prototype represents a batch processing pipeline.
- The process can be automated to run daily, weekly, or monthly depending on business requirements.

#### Output
- The system displays the number of churn vs non-churn customers.
- Model performance metrics (AUC scores) are shown for comparison.
- Predictions can output a churn flag for each customer.

### 4.2 Batch Processing Integration (Optional Feature)
This churn prediction prototype would function as a key analytics element within a larger data ecosystem in a complete end-to-end corporate configuration. The goal is to include the model into a scalable infrastructure that facilitates frequent model retraining, automated data processing, and the smooth supply of churn insights to business processes.  

**How the prototype fits into the full architecture:**

**Data Sources -> AWS S3**
- Customer profiles, transaction history, usage behaviour, and account details are ingested from operational systems.
- These datasets are stored in AWS S3, acting as the centralized data lake.
- S3 provides durability, scalability, and easy integration with analytics tools.

**Batch Processing Layer (Apache Spark)**
- PySpark loads data from S3, cleans it, prepares features, and runs the churn prediction model.
- The prototype code can be scheduled as batch jobs using AWS EMR, AWS Glue, or cron-based workflow managers.
- Batch frequency (daily, weekly, monthly) can be adjusted based on business needs.

**Model Storage and Retraining**
- The trained model can be saved back into S3.
- Retraining jobs can be scheduled in Spark to refresh the model with new customer data.

**Prediction Output Layer**
- Churn predictions are written back into S3 or forwarded to:
  - CRM platforms
  - BI dashboards (Power BI, Tableau, AWS QuickSight)
  - Customer engagement or ticketing systems
- Supports automated identification of high-risk customers.

**Business Decision Layer**
- Marketing and customer service teams can use these predictions for personalised retention strategies.
- The system supports data-driven decisions rather than assumptions.

## 5. Data Processing Workflow
This section outlines how data moves through the system, from initial ingestion to churn prediction, based on the implemented PySpark prototype.

### 5.1 Data Ingestion
- The dataset is loaded from a CSV file (in production, this would be stored in AWS S3).
- PySpark reads the file into a DataFrame for distributed processing.
- Fields include customer age, total purchases, years with the company, number of sites, and a churn label.

### 5.2 Data Cleaning & Preparation
- Missing values are handled using PySpark transformations.
- Irrelevant fields such as customer names, timestamps, and locations are removed when not required.
- Columns are cast into appropriate data types.
- Categorical attributes (e.g., *Company*) are encoded using **StringIndexer** and **OneHotEncoder**.
- The cleaned dataset is prepared with numerical and encoded features ready for modelling.

### 5.3 Feature Engineering
- PySpark’s **VectorAssembler** combines selected variables into a unified feature vector.
- Features included in the prototype:
  - Age  
  - Total Purchase  
  - Years with Company  
  - Number of Sites  
  - Encoded Company field  
- This feature vector serves as input for machine learning algorithms.

### 5.4 Train–Test Split
- Dataset is divided into:
  - **70% Training Data**
  - **30% Test Data**
- Supports robust model evaluation and reduces overfitting risk.

### 5.5 Model Training
- Three machine learning models are trained using PySpark MLlib:
  - Logistic Regression  
  - Random Forest Classifier  
  - Gradient Boosted Trees  
- Each model is fitted on the training data and evaluated on the test dataset.

### 5.6 Model Evaluation
- **Area Under the ROC Curve (AUC)** is the primary evaluation metric.
- **Gradient Boosted Trees** achieved the best performance (AUC ≈ 0.93).

### 5.7 Prediction Output
- Churn predictions are generated (0 = No Churn, 1 = Churn).
- Outputs can be:
  - Saved to AWS S3  
  - Integrated into dashboards  
  - Used for customer segmentation and retention strategies  

### 5.8 Automated Batch Workflow (Future Improvement)
- Currently executed manually.
- Can be automated using:
  - AWS Glue  
  - AWS EMR  
  - Apache Airflow  
  - Cron scheduling  
- Automation allows daily, weekly, or monthly runs without manual involvement.

## 6. Model Development
### 6.1 Model Type
Three models were implemented using PySpark MLlib:
- **Logistic Regression**: Baseline linear classifier for binary outcomes  
- **Random Forest Classifier**: Handles non-linear relationships and feature interactions  
- **Gradient Boosted Trees**: High predictive accuracy and ensemble learning

### 6.2 Training Process
- Models trained on 70% of data with features assembled into a single vector  
- Training performed in batch mode using PySpark MLlib

### 6.3 Hyperparameter Tuning
Key hyperparameters adjusted to improve performance:

- **Logistic Regression**: `regParam`, `maxIter`  
- **Random Forest**: `numTrees`, `maxDepth`  
- **Gradient Boosted Trees**: `maxIter`, `stepSize`, `maxDepth`

## 7. Results
### Accuracy
- Logistic Regression = 0.87  
- Random Forest Classifier = 0.83  
- Gradient Boosted Trees = 0.93  

### Plots
- ROC curves for all models  
- Feature importance plots highlighting top predictors: Total Purchase, Years with Company, Number of Sites

### Interpretation
- Ensemble models outperform linear models  
- Customers with fewer years at the company, lower total purchases, or fewer sites are more likely to churn  
- Prototype can effectively identify high-risk customers for retention strategies

## 8. Discussion
### Strengths
- End-to-end working solution  
- Scalable with PySpark MLlib  
- Gradient Boosted Trees reliable (AUC = 0.93)  
- Modular workflow, easily extendable

### Limitations
- Uses static CSV file, no real-time processing  
- Some features not fully utilized  
- Limited hyperparameter tuning  
- No automated batch or cloud integration

### Proposed Improvements
- Implement automated AWS S3 data ingestion and real-time streaming  
- Apply detailed hyperparameter tuning and cross-validation  
- Explore additional features (customer engagement, support interactions)  
- Integrate orchestration tools (Airflow, AWS Glue) for scheduled retraining

## 9. Future Enhancements
- Automate batch runs with AWS Glue, Airflow, or Cron  
- Integrate Apache Kafka for real-time data ingestion  
- Provide instant predictions for new customer data  
- Deploy on cloud platforms (AWS, Azure) for scalability and accessibility

## 10. License
This prototype uses the open-source dataset and code from [Churn-Detection-using-Spark](https://github.com/ShahedSabab/Churn-Detection-using-Spark) by Shahed Sabab.  
The `customer_churn.csv` dataset and PySpark implementation are adapted for academic purposes.  

All modifications, workflow documentation, and enhancements are original, demonstrating a Big Data solution for customer churn analysis. Refer to the original repository for licensing terms.

