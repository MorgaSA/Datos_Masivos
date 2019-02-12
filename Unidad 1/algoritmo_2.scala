def fun(n:Double): Double ={
    if(n<2){
        return n
    }
    else {
        var m = ((1+math.sqrt(5))/2)
        var j = ((math.pow(m,n)-math.pow((1-m),n))/math.sqrt(5))
        return j
    }
}