def circulo(Area:Double):Double={
    (math.sqrt(Area/3.141516))
}
def Primo(Num1:Int):Int={
    var i = 0
    var a = 0
    var b = 0
    for(i <- 2 to 7){
          var sobra = Num1%i
    if(sobra == 0 && i != 2){
        a = a +1
        println(s"El numero $Num1 no es primo")
    }
    else if(a==0 && sobra == 1){
        println(s"El numero $Num1 es primo") 
    }
  }
    return Num1
}
var bird = "tweet"
println(s"Estoy escribiendo un $bird")

var darthVader = "Hola Luke yo soy tu padre!"
darthVader.slice(5,9)

val NuncaCambies="Nunca"
NuncaCambies="Ni aunque quiera"
var Cambia = "Si como no"
Cambia = "Ya esta"

val Tupla=((2,4,5),(1,2,3),(3.1416,23))
Tupla._3._1

