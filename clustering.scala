// Imports
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row
import org.apache.spark.ml.clustering.KMeans

// TODO: Load the data from the delimited file "wine.csv" into the rdd rawdata
val rawdata = sc.textFile("hdfs:///user/training/mldata/wine.csv")

// TODO: 
// Create an RDD of dense vectors by splitting on commas, casting to float,
// then casting to a dense vector
val vecrdd = rawdata.map(x => x.split(",").map(_.toDouble)).map(x => Vectors.dense(x))

// TODO: Create a case class called line which contains a single element called features
case class line(features:org.apache.spark.mllib.linalg.Vector)

// TODO:
// Convert to a DataFrame with a single column called "features"
// using the case class create above.
val df = vecrdd.map(x => line(x)).toDF()

// TODO: 
// Create a StandardScaler transformer that will
// scale and center the data, taking as input the
// features column and setting the output column
// to be called "scaled"
val ss = new StandardScaler().setWithMean(true).setWithStd(true).setInputCol("features").setOutputCol("scaled")

// TODO:
// Create a KMeans estimator, setting the featuresCol
// to 'scaled' and setting 3 clusters 
val km = new KMeans().setK(3).setFeaturesCol("scaled")

// TODO: 
// Create a pipeline with two stages: the StandardScaler
// transformer and the KMeans estimator created above
val pl = new Pipeline().setStages(Array(ss,km))

// TODO:
// Chain methods of the pipeline (and the resulting pipeline model)
// to create an RDD of predictions called clustered using only one line.
val clustered = pl.fit(df).transform(df)

// TODO: 
// Select only the prediction column, collect all 
// predictions to the driver and print them to screen
clustered.select("prediction").collect().foreach(println)
