# A qr object prints as R prints the list it is.
x <- cbind(a = c(1, 2, 4, 8), b = c(1, 1, 1, 1))
q <- qr(x)
print(q)
q$rank
qr.R(q)
qr.Q(q)
qr.coef(q, c(1, 3, 2, 5))
qr.coef(qr(cbind(u = 1:3, v = 2 * (1:3))), c(1, 2, 4))
