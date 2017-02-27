// Imports
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LassoWithSGD, RidgeRegressionWithSGD, LabeledPoint}

// TODO: 
// Load the data from the delimited file "concrete_train.csv" into the rdd rawdata
val rawdata = sc.textFile("hdfs:///user/training/mldata/concrete_train.csv")

// TODO: 
// Map the RDD of strings to an RDD of dense vectors by splitting 
// on commas, casting each value as floating point, then casting as a dense vector
val vecrdd = rawdata.map(x => x.split(",").map(_.toDouble)).map(x => Vectors.dense(x))

// TODO: 
// Create an instance of StandardScaler, setting withMean and withStd both True. 
// Use it to scale the vector RDD create above, and store in a new RDD called scaled
// Note: Be sure to store the model created by the fit method of the StandardScaler
// in a variable as it will be used in a later step
val ss = new StandardScaler(withMean=true, withStd=true)
val model = ss.fit(vecrdd)
val scaled = model.transform(vecrdd)

// TODO: 
// Create an RDD of labeledPoints using the last value in the list
// as the feature and all but the last value in the list as the features.
// Be sure to explicitly cast the features as a dense vector before
// using in the labeledPoint initialization.
//
// Labeled points are simply a feature vector and then a label.
//
val lprdd = scaled.map{x => 
        val temp = x.toArray
        val label = temp.last
        val features = Vectors.dense(temp.dropRight(1))
        LabeledPoint(label, features)
}

// TODO:
// Create three LinearRegressionWithSGD models 
// by calling the train method with different parameters on the RDD of labeled points.
// The first should have no regularization, the second should use l1 regularization, 
// and the third should use l2 regularization. 
// For both l1 and l2 regularization, set the regParam to 0.2
val lr_model = LinearRegressionWithSGD.train(lprdd, numIterations = 10)
val lr_l1_model = LassoWithSGD.train(lprdd, numIterations = 10, regParam = 0.2, stepSize = 1.0)
val lr_l2_model = RidgeRegressionWithSGD.train(lprdd, numIterations = 10, regParam = 0.2, stepSize = 1.0)

// TODO:
// Create a new RDD of labeled points called "test_lprdd" from the "concrete_test.csv" data
// using the same exact method used to create the "lprdd" RDD above. 
// Be sure to use the same StandardScaler model used to scale the training data, 
// i.e. do not create a new instance of StandardScaler for this step.
val test_rawdata = sc.textFile("hdfs:///user/training/mldata/concrete_test.csv")

val test_vecrdd = test_rawdata.map(x => x.split(",").map(_.toDouble)).map(x => Vectors.dense(x))

val test_scaled = model.transform(test_vecrdd)

val test_lprdd = test_scaled.map{x => 
        val temp = x.toArray
        val label = temp.last
        val features = Vectors.dense(temp.dropRight(1))
        LabeledPoint(label, features)
}

// TODO: 
// Create RDDs of predictions for all three models
val pred_lr = lr_model.predict(test_lprdd.map(x => x.features))
pred_lr.collect
val pred_lr_l1 = lr_l1_model.predict(test_lprdd.map(x => x.features))
pred_lr_l1.collect
val pred_lr_l2 = lr_l2_model.predict(test_lprdd.map(x => x.features))
pred_lr_l2.collect
