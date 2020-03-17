# Python cheat sheet for PySpark

## Import packagess

```python
import findspark
import pyspark
findspark.init()
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
sc =SparkContext()#SparkContext
spark = SparkSession(sc)
sql_sc = SQLContext(sc)
```

## Create spark dataframe from a file

```python
# create a dataframe 
df = spark.read.parquet('file_path')

# register a corresponding query table
df.createOrReplaceTempView('df')
```

```python
# create a pandas dataframe
data = pd.read_csv('file_path')

# convert it into Spark dataframe
df = sql_sc.createDataFrame(data)
```



## Output dataframe and the Schema

```python
df.show()
df.printSchema()
```

## Using SQL in PySpark

```python
spark.sql('select class,count(*) from df group by class').show()
```

## Using PySpark function

```python
df.groupBy('class').count().show()
df.where("City = 'New York'").select("County").distinct().show()
```

## Convert a column in to list

```python
NY_list = s_df.where("City = 'New York'").select('County').distinct().rdd.map(lambda row : row[0]).collect()
```

## Using pixiedust

```python
import pixiedust
from pyspark.sql.functions import col
counts = df.groupBy('class').count().orderBy('count')
display(counts)
```

## ETL process

```python
#import packages for ETL process
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

indexer = StringIndexer(inputCol="class", outputCol="classIndex")
encoder = OneHotEncoder(inputCol="classIndex", outputCol="categoryVec")
vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)
minmaxscaler = MinMaxScaler(inputCol="features_norm", outputCol="scaledFeatures")
pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, normalizer,minmaxscaler])
model = pipeline.fit(df)
prediction = model.transform(df)
prediction.show()
```

---