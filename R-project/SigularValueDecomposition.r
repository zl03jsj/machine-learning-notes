# 创建一个复合变换的矩阵
compose_matrix <- matrix(c(2, 1, 1, 2), nrow = 2, ncol = 2) 
# 取矩阵的特征值特征向量
decompose_matrix <- eigen(compose_matrix)
# 特征值(本征值)
sigular_values <- diag(unlist(decompose_matrix['values']))
# # 特征向量(本征向量)
sigular_vector <- array(unlist(decompose_matrix['vectors']), dim = c(2,2))
# 创建单位为1, 右下角位于(0,0)位置的矩阵
cell_points <- array(c(0, 0, 1, 0, 1, 1, 0, 1), dim=c(2, 4))       

# 绘制对角的向量
plot(x=Nile, y=Nile, lty=3, type = 'l', xlim = c(-3, 5), ylim = c(-3, 5),
     col='lightgray', lwd=1)
axis(1, at=seq(-3, 5, 1))
axis(2, at=seq(-3, 5, 1))
# abline(v=(seq(0,100,25)), col="lightgray", lty="dotted")
# abline(h=(seq(0,100,25)), col="lightgray", lty="dotted") 

library(plotrix)
draw.ellipse(0, 0, sqrt(2), sqrt(2), border='blue', lwd=2)  
grid(nx=10,ny=10,lwd=1,lty=2, col="lightgray") 

lines(x = cell_points[1, c(1:4, 1)], y = cell_points[2, c(1:4, 1)], col="darkgreen", lwd=4)
arrows(length=0.1, col='darkgreen', x0=cell_points[1, 1], y0=cell_points[2, 1], x1 = cell_points[1, 3], y1=cell_points[2, 3], angle= 20, code=2)  

tmp <- t(sigular_vector) %*% cell_points
lines(x = tmp[1, c(1:4, 1)], y = tmp[2, c(1:4, 1)], col='green', lwd=4)
arrows(length=0.1, col='green', x0=tmp[1, 1], y0=tmp[2, 1], x1 = tmp[1, 3], y1 = tmp[2, 3], angle= 20, code=2)  

tmp <- sigular_values %*% tmp
lines(x = tmp[1, c(1:4, 1)], y = tmp[2, c(1:4, 1)], col='yellow2', lwd=2) 
arrows(length=0.1, col='orange', x0=tmp[1, 1], y0=tmp[2, 1], x1 = tmp[1, 3], y1 = tmp[2, 3], angle= 20, code=2)  

tmp <- sigular_vector %*% tmp
lines(x = tmp[1, c(1:4, 1)], y = tmp[2, c(1:4, 1)], col='red4', lwd=5)
arrows(length=0.1, col='yellow1', x0=tmp[1, 1], y0=tmp[2, 1], x1 = tmp[1, 3], y1 = tmp[2, 3], angle= 20, code=2, lwd=6)  

tmp = compose_matrix %*% cell_points 
lines(x = tmp[1, c(1:4, 1)], y = tmp[2, c(1:4, 1)], lty=6, col='blue', lwd=1)
arrows(length=0.1, col='purple3', x0=tmp[1, 1], y0=tmp[2, 1], x1 = tmp[1, 3], y1 = tmp[2, 3], angle= 20, code=2)  

# 绘制用线性变化矩的特征向量来变换cell_points之后的形状
# tmp = sigular_vector %*% cell_points
# lines(x = tmp[1, c(1:4, 1)], y = tmp[2, c(1:4, 1)]) 
# 
# tmp = sigular_values %*% tmp 
# lines(x = tmp[1, c(1:4, 1)], y = tmp[2, c(1:4, 1)]) 
# 
# tmp = t(sigular_vector) %*% tmp 
# lines(x = tmp[1, c(1:4, 1)], y = tmp[2, c(1:4, 1)]) 



# require(grDevices) 
# x <- seq(-10, 10, length= 30)
# y <- x
# f <- function(x, y) { r <- sqrt(x^2+y^2); 10 * sin(r)/r }
# z <- outer(x, y, f)
# z[is.na(z)] <- 1
# op <- par(bg = "white")
# 
# persp(x, y, z, theta = 30, phi = 30, 
#       expand = 0.5, col = drapecol(z))
# persp(x, y, z, theta = 45, phi = 20,
#       expand = 0.5, col = drapecol(z),
#       r=180,
#       ltheta = 120,
#       shade = 0.75, 
#       ticktype = "detailed",
#       xlab = "X", ylab = "Y", zlab = "Sinc( r )" ,
#       #border=30
# ) 