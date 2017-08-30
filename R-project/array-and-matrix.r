source("./zl.utils.r")

show_default_split("array")
my_num <- c(11,22,34,71,14,68,21,22,11,34)

#改变维度
dim(my_num) <- c(2,5)

#切片
my_num[c(1,2,3)] 
c(my_num[1], my_num[4])
my_num[2,]
my_num[6]
my_num[c(1:4, 5:7, 4:6)]

#创建数组
x <- array(10:20, dim=c(2,5))
x
i <- array(c(1:4, 2:6), dim=c(2, 5)) 
i 
dim(i) <- (dim(i)[1] *dim(i)[2])
x[i] 
x[i] = 123

# 数组转换为向量
x <- as.vector(x)


show_default_split("matrix")

mtx <- matrix(c(1:10), 2, 5, TRUE)
mtx
arr <- array(c(1:8))
arr
mtx <- diag(arr)
mtx 
diag(mtx)

# 数组四则运算
show_default_split("array arithmetic")
arr1 <- vector(mode="numeric",length=10)
arr2 <- c(1:10)
print(arr1)
print(arr2)
print(arr1 + arr2)
print(arr1 * arr2)
print(arr1 / arr2)
arr1[] = 1
print(arr1)
print(arr2 / arr1)

print(cbind(arr1, arr2))
print(rbind(arr1, arr2))

show_default_split("matrix arithmetic")
