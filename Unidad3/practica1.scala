//Importaciones necesarias de spark para el programa 
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Generamos la sesion de spark
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

//ensamblamos las columnas
val seleccion = df.select($"Channel",$"Region",$"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
val assembler = new VectorAssembler().setInputCols(Array("Channel","Region","Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

//entrenamos los datos
val traning = assembler.transform(seleccion)
//Se crea un modelo de kmeans
val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(traning)

// Evaluate clustering by calculate Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(traning)
println(s"Within Set Sum of Squared Errors = $WSSSE")
//REsultados
println("Cluster Centers: ")
model.clusterCenters.foreach(println)