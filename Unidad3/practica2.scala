//Importaciones de spark
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Creamos la sesion de spark y el data frame
val spark = SparkSession.builder().getOrCreate()
val df  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")

//Imprimimos el el esquema
df.printSchema()

//Imprimimos la primera linea
df.head(1)

//imprimimos las columnas
val colnames = df.columns

//Imprimimos los primeros datos
val firstrow = df.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}

//creamos una nueva columna com el nombre hora 
val hora = df.withColumn("Hour",hour(df("Timestamp")))

//Creamos una label con los datos de "Clicked on ad"
val logregdataall = (hora.select(df("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age",$"Area Income", $"Daily Internet Usage",$"Male",$"Hour"))
//Eliminamos los datos vacios
val logregdata = logregdataall.na.drop()
//nuevas importaciones para ensamblar los vectores
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
//Ensamblamos las columnas como features
val assembler = (new VectorAssembler().setInputCols(Array("Daily Time Spent on Site", "Age","Area Income", "Daily Internet Usage", "Male", "Hour")).setOutputCol("features"))
//Separamos los datos en dos partes 
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

import org.apache.spark.ml.Pipeline

val lr = new LogisticRegression()

//Preparamos el modelo en el pipeline
val pipeline = new Pipeline().setStages(Array(assembler,lr))
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
