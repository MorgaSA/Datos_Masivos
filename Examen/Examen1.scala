//Verificar solo los numers pares
def Par(num:Int):Boolean={ 
    var par = num % 2
    (par == 0)
}
Par(5)
Par(4)
Par(150)
//Buscar numeros pares en una lista
def listEvens(list:List[Int]): String ={
    for(n <- list){
        if(n%2==0){
            println(s"$n is even")
        }else{
            println(s"$n is odd")
        }
    }
    return "Done"
}

val l = List(1,2,3,4,5,6,7,8)
val l2 = List(4,3,22,55,7,8)
listEvens(l)
listEvens(l2)



//3 7 afortunado
def afortunado(list:List[Int]): Int={
    var res=0
    for(n <- list){
        if(n==7){
            res = res + 14
        }else{
            res = res + n
        }
    }
    return res
}
val af= List(1,7,7)
println(afortunado(af))

//verificar si se puede equilibrar una lista
def balance(list:List[Int]): Boolean={
    var primera = 0
    var segunda = 0

    segunda = list.sum

    for(i <- Range(0,list.length)){
        primera = primera + list(i)
        segunda = segunda - list(i)

        if(primera == segunda){
            return true
        }
    }
    return false 
}

val bl = List(3,2,1)
val bl2 = List(2,3,3,2)
val bl3 = List(10,30,90)

balance(bl)
balance(bl2)
balance(bl3)

//verificar un palindromo
def Cadena(p:String):Boolean={
    var Cad = p 
    var ser = Cad.compareTo(Cad.reverse)
if (ser == 0){
    return true
}
else{
    return false
}
}

Cadena("ANITALAVALATINA")
Cadena("reconocer")
Cadena("Analfabeto")
