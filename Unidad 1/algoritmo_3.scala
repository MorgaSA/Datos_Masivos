def func(x : Int):Int={
    var a = 0
    var b = 1
    var c =0
    var k = 0
    for(k <- 1 to x){
       println(a) 
       c=b+a
        a=b
        b=c
        
    }
    return(x)
}