angle <- (pi / 4)
eigen_ve = matrix(c(2, 1, 1, 2), nrow = 2, ncol = 2)
eigen_ve_t = t(eigen_ve) 
eigen_va = diag(c(2*cos(angle), 2*cos(angle))) 
eigen_ve * eigen_va * eigen_ve_t

rectangle <- array(c())
v <- array(c(1, 0, 0, 1), dim=c(2,2))  

plot(v[1,], v[2,], type = 'l', xlim = c(0, 10), ylim = c(0, 10), col='blue', lwd=1)
rect(xleft = v[1], ybottom = v[2], xright = v[3], ytop = v[4], border = 'red')

tmp <- eigen_ve * v
plot(tmp[1,], tmp[2,], type = 'l', col='green')

grid(nx=10,ny=10,lwd=1,lty=2,col="blue")
axis(1, at=seq(1, 10, 1))
axis(2, at=seq(1, 10, 1))

# 
eigen_ve * eigen_va * eigen_ve_t

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
# 