def func(x : Int):Int={
    var a = 0
    var b = 1
    var k = 0
    for(k <- 1 to x){
        println(a)
        b=b+a
        a=b-a        
    }
    return(x)
}