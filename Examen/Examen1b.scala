import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
df.columns
df.printSchema()
for(row <- df.head(5)){
    println(row)
}
df.describe()
val df2 = df.withColumn("HVRatio", df("High")+df("Volume"))
df2.printSchema()

df2("HVRatio").as("HVR")

df2.select("HVRatio").as("HVR").show
df.orderBy($"High".desc).show()
//Es el valor de las "acciones" para el final de cada dia
df.select(max("Volume")).show()
df.select(min("Volume")).show()

df.filter($"Close"<600).count()

val Per = df.filter($"High">500).count()
val tot = df.select($"High").count()
val rs1 = (Per*100)/tot

df.select(corr("High", "Volume")).show()

df.select(max("High")).show()

val df3 = df.select(year(column("Date"))).distinct()
val df3b = df3.count()
val df4 = df.select("High",distinct(year(column("Date"))).show()
val df5a = df.select(mean("High")).value()
val df5 = df.filter($"High"> df5a )
distinct(year(column("Date"))