import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

df.columns

df.printSchema()

for(row <- df.head(5)){
    println(row)
}

df.describe()

val df2 = df.withColumn("HVRatio", df("Volume")/df("High"))
df2.show()
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

val df3=df.withColumn("Year",year(df("Date")))
val dfmax=df3.groupBy("Year").max()
dfmax.select($"Year",$"max(High)").show()

val dfmonth=df.withColumn("Month",month(df("Date")))
val dfmean=dfmonth.select($"Month",$"Close").groupBy("Month").mean()
dfmean.orderBy($"Month".desc).show()
dfmean.orderBy($"Month").show()