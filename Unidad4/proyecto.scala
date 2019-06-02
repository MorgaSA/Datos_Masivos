//Importaciones de los metodos de clasificacion
//linear suport vector machine
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.ml.classification.LinearSVC
//Deciosion tree
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
//multilayer perceptron
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
//logistic regression
import org.apache.spark.ml.classification.LogisticRegression
//sesion de spark
import org.apache.spark.sql.SparkSession
//para creacion de vectores
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
//limita errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
import org.apache.spark.sql.types.StructType
//Creamos la sesion de spark y el data frame
val spark = SparkSession.builder().getOrCreate()
val df  = spark.read.option("header","true").
option("inferSchema", "true").
option("delimiter", ";").
format("csv").
load("bank-full.csv")
//creamos la etiqueta
//val Etiqueta = new StringIndexer().
//setInputCol("job").
//setOutputCol("label")
//ensamblamos los datos
//,"month","job","marital","education","default","housing","loan","contact","poutcome","y"
//val Ensamble = new VectorAssembler().
//setInputCols(Array("campaign""age","balance","day","duration","pdays","previous")).
//setOutputCol("features")
val logregdata = (df.select(df("campaign").as("label"), $"age", $"balance",$"day", $"pdays",$"previous"))
val assembler = (new VectorAssembler().setInputCols(Array("age","balance","day","duration","pdays","previous")).setOutputCol("features"))
//val data1 = df.select("label","features")

//Separamos los datos en dos partes 
//val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
val splits = logregdata.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)
//------------------------------------------------------------------------------------------
//
//                    suport vector machine
//
//-------------------------------------------------------------------------------------------
//val columns = Array("campaign","age","balance","day","duration","pdays","previous")
//creamos el suport vector machine
//val lsvc = new LinearSVC().
  //setMaxIter(10).
  //setRegParam(0.1).
//setFeaturesCol("features")
//val pipeline = new Pipeline().setStages(Array(Etiqueta,Ensamble, lsvc))
val numIterations = 100
val model = SVMWithSGD.train(training, numIterations)
//entrenamos el modelo
val lsvcModel =lsvc.fit(logregdata)
// Print the coefficients and intercept for linear svc
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
//------------------------------------------------------------------------------------------
//
//                      decision tree
//
//-------------------------------------------------------------------------------------------
//
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))
//
val featureIndexer = new VectorIndexer().
setInputCol("features").
setOutputCol("indexedFeatures").
setMaxCategories(4).
fit(df)
//
val dt = new DecisionTreeRegressor().
setLabelCol("label").
setFeaturesCol("indexedFeatures")
//
val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
// Train model. This also runs the indexer.
val model = pipeline.fit(trainingData)
// Make predictions.
val predictions = model.transform(testData)
// Select example rows to display.
predictions.select("prediction", "label", "features").show(5)
// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator().setLabelCol("label").
setPredictionCol("prediction").setMetricName("rmse")
//root mean square error
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
println(s"Learned regression tree model:\n ${treeModel.toDebugString}")


//------------------------------------------------------------------------------------------
//
//                      multilayer perceptron
//
//-------------------------------------------------------------------------------------------
//separamos los datos en dos grupos
//el de entrenamiento y el de prueba
val splits = df.randomSplit(Array(0.6, 0.4), seed = 1234L)
val trainMP = splits(0)
val testMP = splits(1)
//Creamos nuestra arquitectura 
//la medida de nuestra entrada es de 4
//la medida de las capas intermedias es 5 y 4 
//y la medida de la salida es de 3
val neuronas = Array[Int](4, 5, 4, 3)
//Creamos un trainer con las especificaciones de nuestra clasificacion
val trainer = new MultilayerPerceptronClassifier().
  setLayers(neuronas).
  setLabelCol("label").
  setFeaturesCol("features").
  setPredictionCol("prediction").
  setBlockSize(128).
  setSeed(1234L).
  setMaxIter(100)
//creamos la tuberia de la informacion que queremos la etiqueta y las features 
//asi como nuestro trainer que contiene las especificaciones del modelo
val pipeline = new Pipeline().setStages(Array(Etiqueta,Ensamble,trainer))
//entrenamos el modelo con los datos de nuestra tabla
val model = pipeline.fit(trainMP)
val result = model.transform(testMP)
result.show()
val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels.show()
//medicion de la precision [por medio de multiclassclasificationevaluator
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

//------------------------------------------------------------------------------------------
//
//                      linear regresion
//
//-------------------------------------------------------------------------------------------
//
val lr = new LogisticRegression()

//Preparamos el modelo en el pipeline
val pipeline = new Pipeline().setStages(Array(Ensamble,lr))
//Entrenamos el modelo
val model = pipeline.fit(training)
val results = model.transform(test)


import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// Matriz de confusion
println("Confusion matrix:")
println(metrics.confusionMatrix)

metrics.accuracy
