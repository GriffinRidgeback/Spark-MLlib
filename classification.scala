// Imports
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.DecisionTreeClassifier

// Load the data from the delimited file 'cars_train.csv' into the rdd rawdata
val rawdata = sc.textFile("hdfs:///user/training/mldata/cars_train.csv")

// TODO: 
// Map the RDD of strings to an RDD of Arrays by splitting on commas
val lrdd = rawdata.map(x => x.split(","))

// TODO: 
// Create a new case class for each row of the input data, 
// then create an RDD of these new classes called rowrdd.
// Recall that the rownames are buying, maint, doors, persons, lugboots, safety, label.
// All types should be strings.
// Also recall that the syntax for specifying a case class is
// e.g.,:
// case class line(buying: String, maint: String, ...)
case class line(buying: String, maint: String, doors: String, persons: String, lugboots: String, safety: String, label: String)
val rowrdd = lrdd.map(x => 
    line(x(0), x(1), x(2), x(3), x(4), x(5), x(6))
)

// TODO: Convert rowrdd to a DataFrame, called "train_df"
val train_df = rowrdd.toDF()

// TODO: 
// Create 7 distinct StringIndexer transformers with the outputCol
// parameter set to be the name of the input column appended with the string "_ix"
//
// e.g.,:
// val MyStringIndexer = new StringIndexer.setInputCol("thiscol").setOutputCol("thiscol_ix")
//
// Additionally, store the names of the transformed feature columns 
// (excluding the "label_ix" column) in an Array named "indexedcols"
val indexedcols = Array("buying_ix", "maint_ix", "doors_ix", "persons_ix", "lugboots_ix", "safety_ix")

val si1 = new StringIndexer().setInputCol("buying").setOutputCol("buying_ix")
val si2 = new StringIndexer().setInputCol("maint").setOutputCol("maint_ix")
val si3 = new StringIndexer().setInputCol("doors").setOutputCol("doors_ix")
val si4 = new StringIndexer().setInputCol("persons").setOutputCol("persons_ix")
val si5 = new StringIndexer().setInputCol("lugboots").setOutputCol("lugboots_ix")
val si6 = new StringIndexer().setInputCol("safety").setOutputCol("safety_ix")
val si7 = new StringIndexer().setInputCol("label").setOutputCol("label_ix")

// TODO:
// Create a VectorAssembler transformer to combine all of the indexed
// categorical features into a vector. Provide the "indexedcols" Array created above
// as the inputCols parameter, and name the outputCol "features".
val va = new VectorAssembler().setInputCols(indexedcols).setOutputCol("features")

// TODO:
// Create a DecisionTreeClassifier, setting the label column to your
// indexed label column ("label_ix") and the features column to the 
// newly created column from the VectorAssembler above ("features").
// Store the new StringIndexer transformers, the VectorAssembler, 
// as well as the DecisionTreeClassifier in an Array called "steps"
val clf = new DecisionTreeClassifier().setLabelCol("label_ix")
val steps = Array(si1, si2, si3, si4, si5, si6, si7, va, clf)

// TODO:
// Create a ML pipeline named "pl" using the steps list to set the stages parameter
val pl = new Pipeline().setStages(steps)

// TODO:
// Run the fit method of the pipeline on the DataFrame
// "train_df" to create a pipeline model, and save the
// model in a new variable called "plmodel"
val plmodel = pl.fit(train_df)

// TODO:
// Create a new DataFrame called "test_df" from the cars_test.csv data.
// using the same exact method used to create the "train_df" DataFrame above. 
val test_df = sc.textFile("hdfs:///user/training/mldata/cars_test.csv").map(x => x.split(',')).map(x => 
    line(x(0), x(1), x(2), x(3), x(4), x(5), x(6))
    ).toDF()

// TODO: 
// Run the transform method of the pipeline model created above
// on the "test_df" DataFrame to create a new DataFrame called "predictions"
val predictions = plmodel.transform(test_df)

// Compare the first 15 values in the "prediction" column with those
// rows' values in the "label_ix" column
predictions.select("label_ix", "prediction").show(15)
