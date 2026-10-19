
#ML Pipeline for Airfoil noise prediction
#Data: NASA airfoil noise raw

#Perform ETL, create a Machine Learning Pipeline, evaluate the model and persist the model

#Import libraries
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StandardScaler
import pandas as pd
import os

os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = (
    r"C:\hadoop\bin" + os.pathsep + os.environ["PATH"]
)

#Create a Spark session
spark = SparkSession.builder.appName("Pipeline for Airfoil Noise Prediction").getOrCreate()



#ETL 

#Load the csv file into a dataframe
df = spark.read.csv("NASA_airfoil_noise_raw.csv", header=True, inferSchema=True)
rowcount1 = df.count()

#Drop duplicate rows
df = df.dropDuplicates()
rowcount2 = df.count()

#Drop rows with null values
df=df.dropna()
rowcount3 = df.count()

#Rename column
df = df.withColumnRenamed("SoundLevel", "SoundLevelDecibels")

#Save dataframe in parquet format
df.write.mode("overwrite").parquet("NASA_airfoil_noise_cleaned.parquet")

#Evaluation
print("ETL EVALUATION")
print("Total rows = ", rowcount1)
print("Total rows after dropping duplicate rows = ", rowcount2)
print("Total rows after dropping duplicate rows and rows with null values = ", rowcount3)
print("New column name = ", df.columns[-1])

print("NASA_airfoil_noise_cleaned.parquet exists :", os.path.isdir("NASA_airfoil_noise_cleaned.parquet"))



#CREATE A MACHINE LEARNING PIPELINE

#Load data from NASA_airfoil_noise_cleaned.parquet into a dataframe
df = spark.read.parquet("NASA_airfoil_noise_cleaned.parquet")
rowcount4=df.count()

#Assemble the input columns into a single column named "features"
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")

#Scale the features using StandardScaler
scaler=StandardScaler(inputCol="features", outputCol="scaledFeatures")

#Create a linear regression stage to predict "SoundLevelDecibels"
lr = LinearRegression(featuresCol="scaledFeatures", labelCol="SoundLevelDecibels")  

#Build the pipeline using the above 3 stages
pipeline = Pipeline(stages=[assembler, scaler, lr])

#Split the data into training and testing datasets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

#Fit the pipeline using the training data
pipelineModel = pipeline.fit(train_data)

#Evaluation
print("ML PIPELINE EVALUATION")
print("Total rows = ", rowcount4)
ps = [str(x).split("_")[0] for x in pipeline.getStages()]

print("Pipeline Stage 1 = ", ps[0])
print("Pipeline Stage 2 = ", ps[1])
print("Pipeline Stage 3 = ", ps[2])

print("Label column = ", lr.getLabelCol())



#EVALUATE THE MODEL

#Make predictions on the testing data
predictions = pipelineModel.transform(test_data)

#Print the MSE
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="SoundLevelDecibels", predictionCol="prediction", metricName="mse")    
mse = evaluator.evaluate(predictions)

#Print MAE
evaluator=RegressionEvaluator(predictionCol="prediction",labelCol="SoundLevelDecibels",metricName="mae")
mae = evaluator.evaluate(predictions)

#Print R-squared
evaluator=RegressionEvaluator(predictionCol="prediction",labelCol="SoundLevelDecibels",metricName="r2")
r2 = evaluator.evaluate(predictions)

#Evaluation
print("EVALUATION METRICS")
print("Mean Squared Error = ", round(mse,2))
print("Mean Absolute Error = ", round(mae,2))
print("R Squared = ", round(r2,2))
lrModel = pipelineModel.stages[-1]
print("Intercept = ", round(lrModel.intercept,2))



#PERSIST THE MODEL

#Save the model 
pipelineModel.write().save("Pipeline_Model_Airfoil_Noise_Prediction")

#Load the model from the path "Pipeline_Model_Airfoil_Noise_Prediction"
loadedPipelineModel = PipelineModel.load("Pipeline_Model_Airfoil_Noise_Prediction")

#Make predictions using the loaded model
predictions = loadedPipelineModel.transform(test_data)

#Show the predictions
predictions.select("SoundLevelDecibels", "prediction").show()

#Evaluation
print("MODEL'S COEFFICIENTS")

loadedmodel = loadedPipelineModel.stages[-1]
totalstages = len(loadedPipelineModel.stages)
inputcolumns = loadedPipelineModel.stages[0].getInputCols()

print("Number of stages in the pipeline = ", totalstages)
for i,j in zip(inputcolumns, loadedmodel.coefficients):
    print(f"Coefficient for {i} is {round(j,4)}")


#Stop Spark session
spark.stop()