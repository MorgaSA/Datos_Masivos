//importaciones necesarias para el multilayer perceptor
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
//importaciones para la sesion de spark para leer csv
import org.apache.spark.sql.SparkSession
//importaciones para la creacion de los valores lejibles para un ml
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
//importaciones necesarias para ingresar a las librerias necesarias 
//para quitar muchos errores de el procedimiento de conversion
import org.apache.spark.sql.types._
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
//Empezamos el inicio de sesion de spark
val spark = SparkSession.builder().getOrCreate()
//cargamos el csv
val df = spark.read.option("header","true").option("inferSchema", "true")csv("Iris.csv")
//mostamos el dataframe
df.show()
//imprimimos el esquema
df.printSchema()
//creamos una nueva estructura para el csv con nombres de columna
val Cuchaw = 
StructType(
     StructField("Cinco", DoubleType, true) ::
     StructField("Tres", DoubleType, true) ::
     StructField("Uno", DoubleType, true) ::
     StructField("Cero",DoubleType, true) ::
     StructField("Iris-setosa", StringType, true) :: Nil)
//creamos el nuevo dataframe esta vez con nombres en las columnas
val df2 = spark.read.option("header", "false").schema(Cuchaw)csv("Iris.csv")
df2.columns
//Transformamos el vector a algo que pueda leer el algoritmo de ml
val Etiqueta = new StringIndexer().setInputCol("Iris-setosa").setOutputCol("label")
val Ensamble = new VectorAssembler().setInputCols(Array("Cinco", "Tres", "Uno", "Cero")).setOutputCol("features")
//separamos los datos en dos grupos
//el de entrenamiento y el de prueba
val splits = df2.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
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
val model = pipeline.fit(train)
val result = model.transform(test)
result.show()
val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels.show()
//medicion de la precision [por medio de multiclassclasificationevaluator
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")





