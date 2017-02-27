// Imports
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.ElementwiseProduct
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.stat.Statistics

// TODO:
// Load the raw data into the rawdata RDD
val rawdata = sc.textFile("hdfs:///user/training/mldata/concrete.csv")
rawdata.take(1)

// TODO:
// Convert the rawdata RDD to an RDD of dense vectors called vecrdd.
// Be sure to map the split values to a Double before casting to a dense vector.
// Explanation:
// for each line in rawdata (a pointer to the file in HDFS):
// - split each line at its delimiter creating a group of string values
// - the collection of strings are then each mapped to a Double value
// - the output of the first map is an RDD of arrays of Doubles 
// - the next map takes each array of Doubles and makes a dense vector from it
// - each dense vector is then stored in vecrdd, a collection (RDD) of dense vectors
val vecrdd = rawdata.map(x => x.split(",").map(_.toDouble)).map(x => Vectors.dense(x))
vecrdd.take(1)

// TODO:
// Create a dense vector of weights, naming the vector "weights"
// Make sure that this vector has a weight for each of the 9 columns of the dataset.
val weights = Vectors.dense(Array(0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 1.0))
weights.take(1)

// TODO:
// Create an instance of "ElementwiseProduct"
// Initialize the instance with the weights vector and store it in a variable called "ep"
val ep = new ElementwiseProduct(weights)

// TODO
// Transform vecrdd and store the output in an RDD called "weighted". 
// Then print the first row of vecrdd and the weighted. 
val weighted = ep.transform(vecrdd)
weighted.take(1)
println("Weighted")
weighted.take(1).foreach(println)
println("VecRDD")
vecrdd.take(1).foreach(println)

// TODO:
// Compute basic statistics on the vecrdd using Statistics.colStats()
// Print the mean, variance, and numNonZeros
val stats = Statistics.colStats(vecrdd)
println("Unscaled Statistics")
println(stats.mean)
println(stats.variance)
println(stats.numNonzeros)

val ss = new StandardScaler(withMean = true, withStd = true)
// CallthefitmethodofssonvecrddtocreateaStandardScalerModel, and save the result into a variable called model.
val model = ss.fit(vecrdd)

// To transform the vecrdd, call the transform method of model on vecrdd and save the output to a new RDD called scaled
val scaled = model.transform(vecrdd)
scaled.take(1)

val stats1 = Statistics.colStats(scaled)
println("Scaled Statistics")
println(stats1.mean)
println(stats1.variance)
println(stats1.numNonzeros)
