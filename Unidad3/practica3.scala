//importamos la sesion de spark
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().appName("PCA_Example").getOrCreate()
//cargamos el dataset
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Cancer_Data")
//imprimimos el esquema
data.printSchema()
//importaciones de pca
import org.apache.spark.ml.feature.{PCA,StandardScaler,VectorAssembler}
//importacion de vectores
import org.apache.spark.ml.linalg.Vectors
//seleccionamos las columnas
val colnames = (Array("mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
"mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
"radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error",
"concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius",
"worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity",
"worst concave points", "worst symmetry", "worst fractal dimension"))
//ensamblamos las columnas
val assembler = new VectorAssembler().setInputCols(colnames).setOutputCol("features")
//transformamos el data set utilizando features

val output = assembler.transform(data).select($"features")

//creamos una escalacion de los datos del features
val scaler = (new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false))
//hacemos el entrenamiento
val scalerModel = scaler.fit(output)
//transformamos el modelo entrenado utilizando el output

val scaledData = scalerModel.transform(output)
//creamos el pca
val pca = (new PCA()
  .setInputCol("scaledFeatures")
  .setOutputCol("pcaFeatures")
  .setK(4)
  .fit(scaledData))

val pcaDF = pca.transform(scaledData)

val result = pcaDF.select("pcaFeatures")
result.show()

result.head(1)
