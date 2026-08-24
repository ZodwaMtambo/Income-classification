
from pyspark.sql.functions import col, trim  
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

os.environ["SPARK_LOG_LEVEL"] = "ERROR"


spark = SparkSession.builder.appName("Income Classifier ").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  #removers terminal warning by stoping verbose

df = spark.read.csv("income.csv", header = True, inferSchema= True)

#Data Cleaning 
for column in df.columns:
    df = df.withColumn(column, trim(col(column)))  # remove spaces from every column
    df = df.filter(col(column) != "?")             # remove rows where value is "?"

from pyspark.sql.types import DoubleType

# These are already numbers - no conversion needed
numeric_cols = ["age", "weight", "education_years",
                "capital_gain", "capital_loss", "hours_per_week"]


for c in numeric_cols:
    df = df.withColumn(c, col(c).cast(DoubleType()))
    
print("Total rows after cleaning:", df.count())
df.show(5)

# TEXT columns that need to be converted to numbers
categorical_cols = ["workclass", "education", "marital_status",
                    "occupation", "relationship", "race", "sex", "citizenship"]



indexers = [              #a container with the text columns which will be converted to numbers and sotred in new culumns though iteration
    StringIndexer(inputCol=c, outputCol=c + "_converted", handleInvalid="keep")
    for c in categorical_cols

]

# Y container converting  the <50k values into 1 and 0(we use label for ML  so it knows what to look for to predict)
label_indexer = StringIndexer(inputCol="income_class", outputCol="label", handleInvalid="keep")  #  strin index converts text to numbers , ML only work with numbers not words



# Spark ML requires all input columns to be merged into a single column called "features"
# We use the indexed (number) versions of text columns + the numeric columns
indexed_cols = [c + "_converted" for c in categorical_cols]  
assembler = VectorAssembler(
    inputCols=numeric_cols + indexed_cols,  # combine numeric + converted text columns
    outputCol="features"
)

#data training 
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxBins= 64)
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=10 , maxBins= 64)


#pipeline
rf_pipeline = Pipeline(stages=indexers + [label_indexer, assembler, rf])
dt_pipeline = Pipeline(stages=indexers + [label_indexer, assembler, dt])

print("\nTraining Random Forest...")
rf_model = rf_pipeline.fit(train_data)   # this is where learning happens
 
print("Training Decision Tree...")
dt_model = dt_pipeline.fit(train_data)
 
# predictions
rf_predictions = rf_model.transform(test_data)  # run test data through trained RF model
dt_predictions = dt_model.transform(test_data)  # run test data through trained DT model
 
# accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",           # the actual correct answers
    predictionCol="prediction", # what the model guessed
    metricName="accuracy"       # we want accuracy as our metric
)
 
rf_accuracy = evaluator.evaluate(rf_predictions)
dt_accuracy = evaluator.evaluate(dt_predictions)
 
print("\n========== RESULTS ==========")
print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
print(f"Decision Tree Accuracy: {dt_accuracy:.2%}")
print("==============================")