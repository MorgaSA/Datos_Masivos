var x=0
def fun(n:Int):Int={
    if(n<2){
    return n
}
else{
    return((n -1)+(n+2))
}
}def fun(n:Int):Int={
    if(n<2){
    return n
}
else{
    return((n -1)+(n+2))
}
}


def function(Veces:Int):Int={
    var dato=0
    var Sum = 0
    var dato2 = 1
    do{
        var bandera = Veces
        if(Veces==1){
            return dato
        }
        else if(Veces==2){
            return dato2
        }
        else {
            Sum = dato + dato2
            dato = dato2
            dato2 = Sum
        }
         bandera = bandera -1
    }while(bandera>0)
   
}



