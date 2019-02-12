import scala.collection.mutable.ListBuffer
def fib(n:Int):Int={
    if(n<2){
        return n
    }
    else{
       var almacen = 0
       var VectorMod = new ListBuffer[Int]()
       var k = 0
       var i = 0
        for(i <- 1 to n){
            VectorMod += almacen
            almacen = almacen + 1
        } 
        for(k <- 1 to n+1){
        VectorMod(k)=VectorMod(k-1) + VectorMod(k-2)
         }
        val Vector = VectorMod.toList
        
    }
    return n
}
def printList(args: List[Int]): Unit = {
  args.foreach(println)
}
