//Ejercicios Operaciones
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008.csv")
df.printSchema()
import spark.implicits._

df.select(expm1(column("Volume"))).show()
df.select(factorial(column("Volume"))).show()
df.select(floor(column("Open"))).show()
df.select(hypot(column("High"),column("Low"))).show()
df.select(cbrt(column("Volume"))).show()
df.select(cbrt(column("High"))).show()
df.select(cbrt(column("Low"))).show()
df.select(abs(column("Open"))).show()
df.select(year(column("Date"))).show()
df.select(month(column("Date"))).show()
df.select(last_day(column("Date"))).show()
df.select(bround(column("Open"))).show()
df.select(bround(column("Close"))).show()
df.select(sqrt(column("Volume"))).show()
val Res1 = greatest(column("Open"),column("Close"),column("High"),column("Low"))
val Res2 = df.select(column("Open"),column("Close"),column("High"),column("Low"),Res1)
Res2.show()
df.select(max(column("Volume"))).show()
df.select(min(column("Volume"))).show()
df.select(corr("Open", "Close")).show()
df.filter($"Volume" === 1399500 && $"Open" > 515.3).show()
df.filter($"Volume" < 1399500 && $"Open" > 515.3).show()
val Alm1 = df.filter($"Volume" === 1399500 && $"Open" > 515.3).collect()
val Con1 = df.filter($"Volume" < 1399500 && $"Open" > 515.3).count()

//Ejercicios contain null
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()
val df = spark.read.option("header", "true").option("inferSchema","true")csv("ContainsNull.csv")
df.printSchema()
df.show()

df.na.drop().show()
df.na.drop(4).show()
df.na.fill(0).show()
df.na.fill("Missing Data").show()
df.na.fill("Empty", Array("Name")).show() 
df.na.fill(100, Array("Sales")).show() 
df.na.fill("Caja", Array("Puesto")).show() 
df.na.fill("Nada", Array("Inven")).show() 
df.na.fill(0, Array("Phone")).show()

df.describe().show()
df.na.fill(400.5, Array("Sales")).show()
df.na.fill("Missing name", Array("Name")).show()

val df2 = df.na.fill(400.5, Array("Sales"))
df2.show()
val df3 = df2.na.fill("Missing name", Array("Name"))
df3.show()


