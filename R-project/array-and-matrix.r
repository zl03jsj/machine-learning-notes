source("./zl.utils.r")

show_default_split("array")
my_num <- c(11,22,34,71,14,68,21,22,11,34)

#改变维度
dim(my_num) <- c(2,5)

#切片
my_num[c(1,2,3)] G
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

# matrix translation
mya <- array(1:10, dim=c(2, 5))
print(mya)
print(t(mya))
print("aperm(mya, perm=c(2,1,3))")
print(aperm(mya, perm=c(2,1,3)))

mya <- array(1:24, dim=c(2, 3, 4))
mya
show_default_split('aperm, 这个函数真的有点绕, 只可意会不可言传...')
print(aperm(mya, perm=c(3,1,2))) 

b <- array(c(5:6))

# 内积 transvection(scaler product)
a %o% b
outer(a, b, "*") 

a <- array(1:9, dim=c(3,3))
b <- array(0:8, dim=c(3,3)) 
a %*% b 
crossprod(a, b) 
t(a) %*% b      

# 计算线性方程组的解 Solutions of Linear Equations
# a %*% x = b,下面的示例计算 x, 其中a为系数矩阵; b为常数项, 
# 如果b缺失, 则默认为单位矩阵
a <- array(1:4, dim=c(2,2))
b <- c(5, 8) 
solve(a, b) 

# 矩阵的逆矩阵, 当solve第二个参数为单位矩阵,或第二个参数不存在的时候
# solve就是计算逆矩阵
b <- diag(2) # 创建单位矩阵 
solve(a, b)  
a %*% solve(a)

# 矩阵的特征值和特征向量[本征值和本征向量](eigen value & eigen vector)
a <- array(c(1:16), dim=c(4, 4))
b <- a
# 把b设置为对称矩阵, 
for( i in 1:(dim(b)[1]-1) ) 
    for(j in (i+1) : (dim(b)[1]) )
        b[i, j] = b[j, i]   
# eigen 函数,计算本征值和本征向量, 第二个参数指定, 是否假设matrix为对称矩阵,
# 如果指定为TRUE, 则只使用左下角(包含对角线)的值
# identical函数 比两个矩阵是否相等
identical( eigen(a)["vectors"], eigen(b)["vectors"] )
identical( eigen(a, TRUE)["vectors"], eigen(b)["vectors"] )

# 计算矩阵的行列式
a <- array(1:4, dim=c(2,2))
det(a)

# 计算 矩阵乘以向量组成的矩阵的结果等于
# 矩阵分别乘以向量以后再组成的矩阵
a <- array(1:4, dim=c(2, 2)) 
b <- array(c(2,10), dim=c(2, 1))
c <- array(c(5,6),  dim=c(2, 1)) 
d <- array(c(b, c), dim=c(2, 2)) 
a %*% b
a %*% c 
a %*% d