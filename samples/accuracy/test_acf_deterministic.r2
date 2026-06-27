# Deterministic ACF test: x = 1..10
# Hand-computed: r(0)=1, r(1)=0.7, r(2)=0.412...
x <- c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
a <- acf(x, lag.max = 3)
print(a$acf)
