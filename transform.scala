// Imports
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.{StandardScaler, PolynomialExpansion}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row


// TODO:
// Load the raw data from concrete.csv into the "rawdata" RDD
val rawdata = sc.textFile("hdfs:///user/training/mldata/concrete.csv")

// TODO:
// Split the rawdata rdd on commas, convert the elements to Double 
// and save in an rdd called "splits"
val splits = rawdata.map(x => x.split(",").map(_.toDouble))

// Create a case class called line that will 
// be used to create a DataFrame from splits.
// The case class should have two elements, 
// "label" which should be a Double and "features"
// which should be a dense vector
case class line(label: Double, features: org.apache.spark.mllib.linalg.Vector)

// TODO:
// Use the newly created case class above in a map to map the last
// element of the arrays in "splits" to the "label" field, and the remaining
// elements to the "features" field. Make sure to cast the features to a dense
// vector before using in the case class.
val df = splits.map{x => 
    val label = x.last
    val features = Vectors.dense(x.dropRight(1))
    line(label, features)
}.toDF()

// TODO:
// Create an instance of "StandardScaler" named "ss" to scale
// the "features" column and output this to a new column 
// called "scaledfeatures"
val ss = new StandardScaler().setWithStd(true).setWithMean(true).setInputCol("features").setOutputCol("scaledfeatures")

// TODO: 
// Create an instance of "PolynomialExpansion" named "pe" to transform 
// the "scaledfeatures" column and output this to a new column called 
// "expandedfeatures" (set "degree" = 2 when initializing).
val pe = new PolynomialExpansion().setDegree(2).setInputCol("scaledfeatures").setOutputCol("expandedfeatures")

// TODO:
// Create a Pipeline estimator
// Use "ss" as the first stage and "pe" as the second stage.
val pl = new Pipeline().setStages(Array(ss,pe))

// TODO:
// Call the "fit" method of the "pl" estimator on "df" to create a 
// "PipelineModel" and save it in a variable called "model"
val model = pl.fit(df)

// TODO:
// Create a new DataFrame called "transformed" by calling the "transform"
// method of "model" on "df" and then print the columns using the "columns"
// attribute of the "transformed" dataframe.
val transformed = model.transform(df)
println(transformed.columns)
