//------------------------------------------------------------------------------------------
//
//                    Importaciones
//
//-------------------------------------------------------------------------------------------
  //Para creacion, limpieza y ensamble de los datos  
  import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
  import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
  import org.apache.spark.ml.feature.LabeledPoint
  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.ml.linalg.Vectors
  import org.apache.spark.rdd.RDD
  import org.apache.spark.mllib.evaluation.MulticlassMetrics
  import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
  //para creacion de las tuberias
  import org.apache.spark.ml.Pipeline
  //Para creacion de sesiones de spark
  import org.apache.spark.sql.SparkSession
  //Evita algunos errores
  import org.apache.spark.sql.types._
  import org.apache.log4j._
  Logger.getLogger("org").setLevel(Level.ERROR)
  //Para los diferentes metodos de clasificacion
  //Kmeans
  import org.apache.spark.ml.clustering.KMeans
  //Linear suport vector machine
  import org.apache.spark.ml.classification.LinearSVC
  //Decision tree clasificator
  import org.apache.spark.ml.classification.DecisionTreeClassificationModel
  import org.apache.spark.ml.classification.DecisionTreeClassifier
  import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
  //multilayer perceptron
  import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
  import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
  //Logistic regresion
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.mllib.evaluation.MulticlassMetrics
  import org.apache.spark.ml.evaluation.RegressionEvaluator
  import org.apache.spark.ml.regression.LinearRegression
  //Creamos la sesion de spark
  val spark = SparkSession.builder().getOrCreate()
  //cargamos el csv bank-full
  val datos = spark.read.option("header","true").
  option("inferSchema","true").
  format("csv").
  load("bank-full.csv")
  //mostramos los primeros 10 datos del dataframe
  datos.show(10)
  //creamos una serie de columnas para la columna "y" que es de tipo string 
  //la convertimos simultaneamente a int con condiciones
  val columna1 = datos.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
  val columna2 = columna1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
  val columna3 = columna2.withColumn("y",'y.cast("Int"))
  //Seleccionamos las columnas a utilizar para los siguentes modelos
  val featureCols = Array("age","balance")
  //Ensamblamos las columnas con un vectorAssembler para hacer los features
  val assembler = new VectorAssembler().
  setInputCols(featureCols).
  setOutputCol("features")
  //transformamos el assembler con la colimna de "y"
  val df = assembler.transform(columna3)
  df.show(5)
  //Renombramos la columna "y" como label
  val change = df.withColumnRenamed("y", "label")
  val ft = change.select("label","features")
  //mostramos datos
  ft.show()
//------------------------------------------------------------------------------------------
//
//                      Croos validation
//
//--------------------------------------------------------------------------------------------
//creamos un linar regresion model
val lr = new LinearRegression().setMaxIter(10)
//creamos una matris de parametros
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).
addGrid(lr.fitIntercept).addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).build()
//creamos un train para los datos del linear regresion
val trainValidationSplit = new TrainValidationSplit().setEstimator(lr).setEvaluator(new RegressionEvaluator).
setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)
//Ensamblamos para crear el features
val assembler = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).
setOutputCol("features"))
//ensamblamos para crear el label
val assembler1 = (new VectorAssembler().setInputCols(Array("y")).setOutputCol("label"))
//creamos un dataframe con los datos del asembler tranformando la columna
val df = assembler.transform(columna3)
//mostramos
df.show(5)
//transformamos con el df
val df1 = assembler1.transform(df)
//mostramos
df1.show(5)
//creamos una nueva columna para label 
val  c55= df1.withColumn("label",'y.cast("Double"))
//creamos un arreclo para separar el training y el test
val Array(trainingData, testData) = c55.randomSplit(Array(0.7,0.3))
//entrenamos con el training
val model22 = trainValidationSplit.fit(trainingData)
//tranformamos el trainig con el test y seleccionamos el features y el label
 model22.transform(testData).select("features", "label").show()
//------------------------------------------------------------------------------------------
//
//                      kmeans
//
//------------------------------------------------------------------------------------------
//creamos un modelo kmeans
val kmeans = new KMeans().setK(7).setSeed(1L).setPredictionCol("prediction")
//creamos el modelo del trainig
val model = kmeans.fit(df)
//creamos una web service security especification
val WSSE = model.computeCost(df)
//imprimimos
println("Cluster Centers: ")
k_model.clusterCenters.foreach(println)
//creamos categorias transformando el modelo con el test
val categories = model.transform(testData)
//seleccionamos columnas
val mostrandoresult =categories.select($"age",$"balance",$"prediction").
groupBy("age","balance","prediction").
agg(count("prediction")).orderBy("age","balance","prediction")
//mostramos datos
mostrandoresult.show(5)
categories.show(5)
//------------------------------------------------------------------------------------------
//
//                    suport vector machine
//
//-------------------------------------------------------------------------------------------
//creamos una serie de columnas para la columna "label"  
//la convertimos simultaneamente de 1 a 0 y de 2 a 1 con condiciones
val cs1 = ft.withColumn("1abel",when(col("label").equalTo("1"),0).otherwise(col("label")))
val cs2 = cs1.withColumn("label",when(col("label").equalTo("2"),1).otherwise(col("label")))
val cs3 = cs2.withColumn("label",'label.cast("Int"))
//Creamos el metodo del linear suport vector machine
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
//entrenamos el modelo con nuestros datos
val lsvcModel = lsvc.fit(cs3)
//imprimimos los coheficientes
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
//------------------------------------------------------------------------------------------
//
//                      decision tree clasifier
//
//------------------------------------------------------------------------------------------
//creamos el lableindexer utilizando el label
val labelIndexer = new StringIndexer().
  setInputCol("label").
  setOutputCol("indexedLabel").
  fit(cs3)
//Creamos el fratures indexer con el features
val featureIndexer = new VectorIndexer().
  setInputCol("features").
  setOutputCol("indexedFeatures").
  setMaxCategories(4).
  fit(cs3)
// separamos los datos en training y test para mantener un 70% para el training y el 30% para el test
val Array(trainingData1, testData1) = cs3.randomSplit(Array(0.7, 0.3))
//Entrenamos el modelo del decision tree
val dt = new DecisionTreeClassifier().
  setLabelCol("indexedLabel").
  setFeaturesCol("indexedFeatures")
// convertimos los index label en lables normales
val labelConverter = new IndexToString().
  setInputCol("prediction").
  setOutputCol("predictedLabel").
  setLabels(labelIndexer.labels)
//encadenamos los indexers en el pipeline
val pipeline = new Pipeline().
setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
//entrenamos el modelo
val model = pipeline.fit(trainingData1)
//hacemos las predicciones
val predictions = model.transform(testData1)
//seleccionamos las columnas que necesitamos
predictions.select("predictedLabel", "label", "features").show(5)
// seleccionamos (prediction, true label) y revisamos el error
val evaluator = new MulticlassClassificationEvaluator().
  setLabelCol("indexedLabel").
  setPredictionCol("prediction").
  setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
//imprimimos el arbol
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)
//------------------------------------------------------------------------------------------
//
//                      multilayer perceptron
//
//-------------------------------------------------------------------------------------------
//separamos los datos para el modelo
val splits = categories.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
//Creamos nuestra arquitectura 
//la medida de nuestra entrada es de 4
//la medida de las capas intermedias es 5 y 4 
//y la medida de la salida es de 3
val layers = Array[Int](4, 5, 4, 3)
//creamos el modelo para el multilayer perceptron
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
//creamos las predicciones utilizando el las celulas del kmeans y el predictions del kmeans
val  c4= categories.withColumn("prediction",'prediction.cast("Double"))
//creamos una columna para el label para la prediccion
val  c5= c4.withColumn("label",'prediction.cast("Double"))
//seleccionamos las columnas prediction y label
val predictionAndLabels = c5 .select("prediction", "label")
//evaluamos el accuracy 
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
//------------------------------------------------------------------------------------------
//
//                      logistic regresion
//
//-------------------------------------------------------------------------------------------
//creamos un dataframe convertimos y a label y seleccionamos las columnas a usar
val logregdataall = (columna3.select(columna3("y").as("label"), $"age",$"balance",$"day",$"duration",$"pdays",$"previous"))
//Ensamblamos las columnas como features
val ensamblador = (new VectorAssembler().setInputCols(Array("age","balance","day","duration","pdays","previous")).setOutputCol("features"))
//Separamos los datos en dos partes 
val Array(training1, test1) = logregdataall.randomSplit(Array(0.7, 0.3), seed = 12345)
//creamos el metodo de logistic regresion
val lr = new LogisticRegression()
//Preparamos el modelo en el pipeline
val pipeline1 = new Pipeline().setStages(Array(ensamblador,lr))
//Entrenamos el modelo
val modelo = pipeline1.fit(training1)
val resultados = modelo.transform(test1)
val prediccionAndLabels = resultados.select($"prediction",$"label").as[(Double, Double)].rdd
val metricas = new MulticlassMetrics(prediccionAndLabels)
// Matriz de confusion
println("Confusion matrix:")
println(metricas.confusionMatrix)
//revisamos el accuracy
metricas.accuracy


