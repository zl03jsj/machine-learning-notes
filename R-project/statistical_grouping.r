source("./zl.utils.r")

#--------因子(factor)--------
my_num <- c(11, 22, 34, 71, 14, 68, 21)
nums <- factor(my_num)
nums 
nums <- as.factor(my_num)
nums
levels(nums)

ordered(nums)

age <- c(25, 12, 15, 12, 25)
ordered(age)

score <- c(88, 85, 75, 97, 92, 77, 74, 70, 63, 97)
cut(score, breaks = 3) 

#  标准差计算函数
standard_deviation <- function(samples) {
  m <- mean(samples)
  total <- 0
  for( price in samples) {
    total <- total + (price - m) ^ 2
  }
  return(total ^ (1/2))
}


fruit_class <- c("苹果","梨子","橘子","草莓","苹果","橘子","橘子","草莓","橘子","草莓") 
fruit_pricess <- c(3.5,2.5,1.5,5.5,4.2,3.2,2.8,4.8,2.9,5.8)


# 计算水果平均价格(算术平均)
tapply(fruit_pricess, fruit_class, mean)
# 计算最便宜的水果(最小值)
tapply(fruit_pricess, fruit_class, min)
# 计算相同水果价格的离散度(标准差)
tapply(fruit_pricess, fruit_class, sd)

# 标准误
# 最常见的标准误是平均值的标准误
# 对一个总体多次抽样，每次样本大小都为n，那么每个样本都有自己的平均值，这些平均值的标准差叫做标准误差
# 标准差是单次抽样得到的，用单次抽样得到的标准差可以估计多次抽样才能得到的标准误差
stderr <- function(x) sqrt(var(x)/length(x))
tapply(fruit_pricess, fruit_class, stderr)

