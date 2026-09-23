# Ardon-R2 Function Reference — 438 Built-in Functions

## Core (83)
```
seq(from,to,by=)   Regular sequence; also seq(along=x), seq_len(n)
rep(x,times=,each=)   Repeat a vector
rev(x)             Reverse a vector
sort(x,decreasing=)   Sort a vector
unique(x)          Distinct values, first occurrence order
which(x)           Indices where a logical vector is TRUE
which.max(x) / which.min(x)   Index of the first maximum / minimum
all(x) / any(x)    TRUE if all / any elements are TRUE (na.rm= honoured)
isTRUE(x) / isFALSE(x)   TRUE only for a length-1, non-NA logical of that value
nrow(x) / ncol(x)  Row / column count of a matrix or data frame
rowSums(x) / colSums(x)     Row / column totals of a matrix
rowMeans(x) / colMeans(x)   Row / column means
t(x)               Matrix transpose
exists(name)       TRUE if a variable of that name is defined
ls()               Names of objects in the current environment
rm(x)              Remove objects from the environment
stop(msg)          Raise an error (caught by tryCatch)
warning(msg)       Raise a warning
message(msg)       Print a message to the console (stderr in R)
as.data.frame(x)   Coerce a matrix or list to a data frame
as.single(x) / is.single(x)   Single-precision marker (R compatibility; values stay double)
c(...)              Create vector: c(1,2,3)
isTRUE(x)/isFALSE(x) Test for a length-1 logical TRUE/FALSE
identical(a,b)      Exact (type-strict) equality
all.equal(a,b)      Near-equality (numeric tolerance) — TRUE or a diff message
diag(x)             Matrix diagonal / diagonal matrix / k×k identity
length(x)           Length of vector
print(x)            Print value
cat(...)            Print without newline
clear() / cls()     Clear the console (GUI buffer / terminal)
typeof(x)           Type name: "numeric", "character", etc.
class(x)            Class of object
list(...)           Create named list
data.frame(...)     Create data frame: data.frame(x=1:3, y=c("a","b","c"))
matrix(data,nrow,ncol) Create matrix
as.numeric(x)       Convert to numeric
as.character(x)     Convert to character
as.integer(x)       Convert to integer
as.logical(x)       Convert to logical
as.factor(x)        Convert to factor
is.na(x)            Test for NA (TRUE for NaN too, as in R)
is.nan(x)           Test for NaN (FALSE for NA)
is.infinite(x)      Test for Inf/-Inf
is.finite(x)        Test for finite values (NA/NaN/Inf are FALSE)
is.numeric(x)       Test if numeric
is.character(x)     Test if character
is.logical(x)       Test if logical
is.null(x)          Test if NULL
is.data.frame(x)    Test if data.frame
is.factor(x)        Test if factor
is.ordered(x)       Test if ordered factor
is.matrix(x)        Test if matrix
TRUE/FALSE/T/F      Logical constants
NA                  Missing value
NULL                Null value
seq_len(n)          1:n (length-safe; empty when n=0)
seq_along(x)        1:length(x)
unlist(x)           Flatten a list to an atomic vector
setNames(x,nm)      Return x with names set
append(x,vals,after) Insert values into a vector
invisible(x)        Return x without auto-printing
local(expr)         Evaluate expr in a fresh local environment
missing(arg)        TRUE if the parameter was not supplied in the call
switch(expr,...)    Select a branch by name or position
with(data,expr)     Evaluate expr in data's column/element scope
stopifnot(...)      Error unless all arguments are TRUE
attr(x,which)       Get one attribute (names/class/dim/custom)
attributes(x)       Get all attributes as a named list
structure(x,...)    Return x with attributes set (class=, names=, dim=, …)
inherits(x,what)    TRUE if `what` is in class(x)
format(x,nsmall=)   Format values as character
numeric(n)          Zero numeric vector of length n
integer(n)          Zero integer vector
character(n)        Empty-string vector
logical(n)          FALSE vector
as.matrix(x)        Coerce data.frame/vector to a matrix
as.vector(x)        Strip attributes to a plain vector
as.list(x)          Coerce to a list (df → list of columns)
is.function(x)      Test if a function (closure)
is.list(x)          Test if a list
is.vector(x)        Test if an atomic vector
is.element(el,set)  el %in% set
tryCatch(expr,error=,finally=)  Function-form error handling
match.arg(arg,choices)  Match arg against choices (exact / unique prefix)
nargs()             Number of arguments the enclosing call received
```

## Math (48)
```
abs(x)      Absolute value          sqrt(x)     Square root
round(x,n)  Round to n digits       log(x)      Natural log
exp(x)      Exponential             ceiling(x)  Round up
floor(x)    Round down              cumsum(x)   Cumulative sum
cumprod(x)  Cumulative product      cummax(x)   Cumulative max
cummin(x)   Cumulative min          diff(x)     Differences
prod(x)     Product of all          range(x)    Min and max
max(x)      Maximum                 min(x)      Minimum
sum(x)      Sum of all              sign(x)     Sign (-1,0,1)
signif(x,d) Round to d significant figures
pmin(...)   Parallel (element-wise) minimum of several vectors
pmax(...)   Parallel (element-wise) maximum
factorial(x) x!  (via gamma)        gamma(x)    Γ(x)
lgamma(x)   log Γ(x)                beta(a,b)   Beta function
choose(n,k) Binomial coefficient    combn(x,m)  m-combinations → matrix
trunc(x)    Truncate toward zero    log2(x)     Base-2 log
log10(x)    Base-10 log             log1p(x)    log(1+x), accurate near 0
expm1(x)    exp(x)-1, accurate near 0
sin(x)      Sine (radians)          cos(x)      Cosine
tan(x)      Tangent                 asin(x)     Arcsine
acos(x)     Arccosine               atan(x)     Arctangent
atan2(y,x)  Two-argument arctangent, quadrant-aware
sinh(x)     Hyperbolic sine         cosh(x)     Hyperbolic cosine
tanh(x)     Hyperbolic tangent
crossprod(x,y)  t(x) %*% y without forming the transpose (y defaults to x)
outer(x,y,FUN)  Outer product matrix; FUN defaults to "*", any binary function works
tensor(data,shape=)  n-dimensional array constructor
uniroot(f,lower,upper)   Find a root of f by bisection (→ $root)
integrate(f,lower,upper) Numerical integral of f (Simpson) (→ $value)
optimize(f,lower,upper)  Golden-section minimize/maximize (→ $minimum)
```

## Statistics (49)
```
cov(x,y)    Sample covariance (errors on unequal lengths, as R)
dbinom(x,size,prob) / pbinom(q,size,prob)   Binomial density / CDF
dpois(x,lambda) / ppois(q,lambda)           Poisson density / CDF
dexp(x,rate) / pexp(q,rate) / qexp(p,rate)  Exponential density / CDF / quantile
dchisq(x,df) / pchisq(q,df) / qchisq(p,df)  Chi-squared density / CDF / quantile
dt(x,df) / pt(q,df) / qt(p,df)              Student-t density / CDF / quantile
binomial() / poisson() / gaussian()         glm family objects (family= argument)
mean(x)         Arithmetic mean
sd(x)           Standard deviation
var(x)          Variance
cor(x,y)        Correlation
median(x)       Median
quantile(x,p)   Quantile at probability p
lm(y~x,data)    Linear regression
glm(y~x,data,family) Generalized linear model
aov(y~x,data)   One-way analysis of variance
                Repeated measures:
                  aov(y ~ x + Error(subject), data=df)
                  aov(y ~ x + Error(subject/treatment), data=df)
                Multi-stratum output matching R's summary(aov).
                Bit-identical to R's output when R uses factor(subject).
anova(model)    ANOVA table from a fitted model
t.test(x,mu)    T-test, paired and unpaired forms
                One-sample:    t.test(x, mu=0)
                Two-sample:    t.test(x, y)
                Welch:         t.test(x, y) [default for unpooled]
                Paired:        t.test(x, y, paired=TRUE)
                Formula:       t.test(y ~ group, data=df)
                Paired w/ id:  t.test(y ~ group, id=subj, paired=TRUE)
                R2 extensions (not supported by R itself):
                  t.test(y ~ group + Error(subject), paired=TRUE, data=df)
                  t.test(y ~ Error(subject), paired=TRUE, data=df)
                    (pairs each subject's 2 obs by row order)
chisq.test(x)   Chi-squared test
hotelling.test  Multivariate Hotelling's T²
                One-sample:    hotelling.test(X)
                With null mu:  hotelling.test(X, mu=c(0,0,0))
                Two-sample:    hotelling.test(X, Y)
                Paired/RM:     hotelling.test(X, Y, paired=TRUE)
                X and Y are n×p matrices of multivariate observations.
                Returns T², F, df, p-value.
manova(formula,data) Multivariate ANOVA
                LHS is a multivariate response (use cbind):
                  manova(cbind(y1, y2, y3) ~ group, data=df)
                Reports four classical statistics:
                  Wilks' Lambda (with Rao F-approximation + p-value)
                  Pillai's trace
                  Hotelling-Lawley trace
                  Roy's largest root
                Returns TypeInstance with all four + eigenvalues vector.
confint(model, level=0.95)  Coefficient confidence intervals as a p x 2 matrix (t quantile for lm, normal for glm)
rnorm(n)        Random normal
runif(n)        Random uniform
rbinom(n,size,prob) Random binomial
rpois(n,lambda) Random Poisson
dnorm(x)        Normal density
pnorm(x)        Normal CDF
qnorm(p)        Normal quantile
set.seed(n)     Set random seed
sample(x,n)     Random sample
mad(x)          Median absolute deviation (constant=1.4826)
fivenum(x)      Tukey's five-number summary
dexp/pexp/qexp(x, rate=)         Exponential density / CDF / quantile
dbinom/pbinom(x, size, prob)     Binomial pmf / CDF
dpois/ppois(x, lambda)           Poisson pmf / CDF
dt/pt(x, df)                     Student-t density / CDF
dchisq/pchisq(x, df)             Chi-squared density / CDF
pf(q, df1, df2)                  F-distribution CDF
qt/qchisq(p, df)                 t / chi-squared quantiles
qf(p, df1, df2)                  F quantile
qbinom(p, size, prob)            Binomial quantile
qpois(p, lambda)                 Poisson quantile
rexp(n, rate)                    Exponential random variates
density(x)                       Gaussian kernel density estimate (list $x/$y)
```

## Dates & Time Series (54)
```
as.Date(s)         Parse a date string → Date (days since 1970-01-01)
as.POSIXct(s)      Parse a date-time string → POSIXct (seconds since epoch)
format(d, fmt)     Render a Date/POSIXct, e.g. format(d,"%Y/%m/%d"); strftime alias
strftime(d, fmt)   Same as format() for dates
Sys.Date()         Current date (class Date)
Sys.time()         Current date-time (class POSIXct; seconds since 1970-01-01 UTC)
format.Date(d,fmt) / format.POSIXct(d,fmt)   Method form of format() for dates
difftime(a,b)      Difference between two dates (Date − Date also works)
ts(x, start=, frequency=)  Build a regular time series
frequency(x)       Observations per cycle of a ts
start(x)           First time point of a ts, as c(cycle, position)
end(x)             Last time point of a ts
cycle(x)           Position within the cycle for every observation
time(x)            Time of every observation as a numeric vector
tsp(x)             The (start, end, frequency) triple
deltat(x)          Time step between observations (1/frequency)
window(x,start=,end=)   Sub-series between two time points
acf(x, lag.max) / pacf(x, lag.max)   Auto- / partial-autocorrelation
diff(x, lag=, differences=)          Lagged differences
decompose(x)       Classical additive/multiplicative decomposition
lag(x, k)          Shift a series by k observations
is.ts(x)           TRUE for a ts object
is.regular(x)      TRUE when an xts index is evenly spaced
periodicity(x)     Detected spacing of an xts index (daily, weekly, ...)
diff_ts(x,lag=)    Differences of a ts, keeping the ts attributes
xts(x, order.by=)  Irregular time series indexed by Date/POSIXct
index(x)           The time index of an xts
coredata(x)        The values of an xts, index stripped
is.xts(x)          TRUE for an xts object
xts.subset(x,"2024-03")   Rows whose index falls in an ISO date/month/year string
merge.xts(a,b)     Join two xts on their index (outer; NA where absent)
first(x) / last(x)         First / last observation of an xts
na.locf(x)         Forward-fill NAs with the last observation carried forward
to.daily(x)        Aggregate an xts to one row per day (open/high/low/close)
to.weekly(x)       Aggregate to one row per week
to.monthly(x)      Aggregate to one row per month
to.quarterly(x)    Aggregate to one row per quarter
to.yearly(x)       Aggregate to one row per year
apply.daily(x,FUN)     Apply FUN to each day's observations of an xts
apply.weekly(x,FUN)    Apply FUN to each week
apply.monthly(x,FUN)   Apply FUN to each month
apply.quarterly(x,FUN) Apply FUN to each quarter
apply.yearly(x,FUN)    Apply FUN to each year
rollmean(x,k)      Rolling mean over a window of k observations
rollsum(x,k)       Rolling sum
rollsd(x,k)        Rolling standard deviation
rollmax(x,k)       Rolling maximum
rollmin(x,k)       Rolling minimum
tithi(date)        Hindu lunar day: list(tithi 1..30, name, paksha, masa) for a Date
hindu.date(date)   Hindu calendar breakdown of a Date (lunar month, paksha, tithi)
hnc.date(date)     Hindu National Calendar form SSSS-MM-P-TT (Saka year, month, paksha, tithi; "A" marks Adhik Maas)
```

## Machine Learning (23)
```
rpart(y~.,data)         Decision tree (CART)
rf(y~.,data,ntrees)     Random forest
gbm(y~.,data,ntrees)    Gradient boosted trees
kmeans(x,centers)       K-means clustering
knn(train,test,labels,k) K-nearest neighbors
naive.bayes(x,y)        Gaussian naive Bayes
prcomp(x)               Principal component analysis
svd(x)                  Singular value decomposition
eigen(x)                Eigenvalue decomposition (symmetric -> dsyev with
                        vectors; non-symmetric -> dgeev; complex spectra
                        expose $imaginary, $vectors = NULL)
solve(a[, b])           Matrix inverse, or solve a x = b
det(a)                  Determinant (LU)
backsolve(r, x, k, upper.tri=TRUE, transpose=FALSE)   Upper-triangular solve
forwardsolve(l, x, k, upper.tri=FALSE, transpose=FALSE) Lower-triangular solve
rcond(a)                Reciprocal 1-norm condition number (exact, via inverse)
kappa(z)                2-norm condition number sigma_max/sigma_min (exact SVD)
mmap.write(x, path)     Write a numeric vector as a packed-f64 file
mmap.col(path)          Open a memory-mapped column (out-of-core, larger
                        than RAM); sum/mean/sd/var/prod/min/max/range/length
                        stream over the mmap with bounded memory.
                        median/quantile use a streaming two-pass histogram
                        (approximate, bounded memory)
mmap.map(path,FUN,out)  Out-of-core scalar map: stream a transform
                        (log/log2/log10/exp/sqrt/abs/square/neg) over an mmap
                        column to a new file (>RAM in → >RAM out)
mmap.csv(file, sep=",") Out-of-core CSV import: stream the file row-by-row
                        and write each NUMERIC column to its own packed-f64
                        sidecar; returns a named list of mmap.col handles
                        (run sum/mean/sd/... on each, larger than RAM)
mmap.lm(data,response,predictors)
                        Out-of-core least squares: accumulate XᵀX and Xᵀy in
                        one streaming pass over the mmap columns, then solve
                        the normal equations. data = mmap.csv list; response
                        and predictors name its columns. Returns named
                        coefficients ((Intercept), predictors…)
scale(x)                Center and scale
cv(x,y,model,k)         K-fold cross-validation
confusion.matrix(pred,actual) Confusion matrix + F1
```

## Data Handling (35)
```
subset(df, cond)   Rows where cond is TRUE; cond may name columns directly (NSE)
transform(df, z = x + y)   Add or overwrite columns from expressions over the frame
list.meta(x)       Shape summary of a list: $lens, $total_work, $homogeneous (for choosing parallel strategy)
head(x,n)           First n rows
tail(x,n)           Last n rows
str(x)              Structure of object
summary(x)          Summary statistics
names(x)            Column names
dim(x)              Dimensions
nrow(x)/ncol(x)     Row/column count
filter(df,mask)     Keep rows where TRUE
select(df,cols)     Keep specified columns
arrange(df,col)     Sort by column
mutate(df,col=val)  Add/modify column
merge(x,y,by)       Join data frames
rbind(x,y)          Stack rows
cbind(x,y)          Stack columns
order(x)            Sorting indices
rank(x)             Ranks
duplicated(x)       Find duplicates
na.omit(x)          Remove NAs
complete.cases(x)   Rows without NAs
ifelse(test,yes,no) Vectorized if
table(x)            Frequency table
factor(x)           Create factor
levels(f)           Factor levels
nlevels(f)          Number of levels
colnames(x)         Column names
rownames(x)         Row names
data(name)          Load built-in dataset
cut(x,breaks)       Bin a numeric into interval factors
split(x,f)          Split a vector into a list grouped by factor f
setdiff(x,y)        Set difference
union(x,y)          Set union
intersect(x,y)      Set intersection
ave(x,g,FUN=)       Group-wise statistic broadcast back over x
```

## String Functions (20)
```
gregexpr(pattern,x)   All match positions of a regex in each string
regmatches(x,m)    The matched substrings for positions from regexpr/gregexpr
paste(...,sep)      Concatenate with separator
paste0(...)         Concatenate without separator
toString(x,sep)     Collapse x to one comma-separated string
grep(pat,x)         Find pattern (indices)
grepl(pat,x)        Find pattern (logical)
gsub(pat,rep,x)     Replace all matches
sub(pat,rep,x)      Replace first match
substr(x,start,end) Substring
strsplit(x,split)   Split string
nchar(x)            String length
toupper(x)          To uppercase
tolower(x)          To lowercase
trimws(x)           Trim whitespace
startsWith(x,pre)   Starts with prefix
endsWith(x,suf)     Ends with suffix
sprintf(fmt,...)    Formatted string
regexpr(pat,x)      Find match position
substring(x,first,last) Substring (vectorized first/last)
```

## Apply Family (11)
```
mapply(FUN, ...)   Apply FUN element-wise over several vectors in parallel
vapply(x,FUN,FUN.VALUE)   sapply with a declared result type and length
sapply(x,fun)       Apply and simplify
lapply(x,fun)       Apply and return list
apply(x,margin,fun) Apply over matrix margins
tapply(x,idx,fun)   Apply by groups
aggregate(x,by,fun) Aggregate by groups; formula form: aggregate(cbind(y1,y2) ~ g1+g2, data=df, FUN)
do.call(fun,args)   Call function with arg list
Reduce(f,x,init)    Left-fold a binary function over x
Filter(f,x)         Keep elements where f(x) is TRUE
Map(f,...)          Apply f element-wise across vectors → list
```

## I/O (14)
```
read.csv(file, header=TRUE, sep=",", check.names=TRUE)   Read a CSV; header names go through make.names (a b -> a.b, 1x -> X1x, duplicates .1/.2) unless check.names=FALSE
make.names(x, unique=FALSE)   R's valid-name rule: X prefix for a bad start, invalid characters to ".", reserved words get "."; unique=TRUE appends .1, .2 to duplicates
read.parquet(file)  Read a Parquet file as a data.frame (pure-Rust via the
                    parquet/arrow crates; reads row-group by row-group so
                    large files import with bounded memory. Numeric types →
                    numeric, boolean → logical, strings → character)
write.csv(x,file)   Write CSV file
read.table(file)    Read delimited file
write.table(x,file) Write delimited file
read.delim(file)    Read tab-delimited
source(file)        Run R2 script
save(file)          Save session
load(file)          Load session
file.exists(path)   Check if file exists
list.files(path)    List directory
readLines(path,n=)  Read text file lines into a character vector
writeLines(text,con=) Write a character vector as lines (file or console)
```

## Graphics (39)
```
plot.new()         Start a new, empty plot frame
plot.window(xlim,ylim)   Set the coordinate system of the current frame
dev.new()          Open a new graphics device
dev.cur()          Number of the current device
dev.set(n)         Make device n current
dev.list()         Open device numbers
save.plot(file)    Save the current plot to file (.svg/.png/.pdf by extension)
rgb(r,g,b)         Colour from components in 0..1 → "#RRGGBB"
hsv(h,s,v)         Colour from hue, saturation, value in 0..1
gray(level) / grey(level)   Grey level 0 (black) .. 1 (white)
adjustcolor(col,alpha=)     Add transparency to a colour
rainbow(n)         n evenly spaced hues
heat.colors(n) / terrain.colors(n) / topo.colors(n) / cm.colors(n)   R's standard palettes
plot(x,y)         Scatter/line plot; inline params col=/cex=/pch=/type=/lwd=/las=
hist(x)           Histogram
boxplot(x)        Box-and-whisker
barplot(x)        Bar chart
pairs(df)         Scatterplot matrix of a data.frame / matrix
matplot(x,y)      Plot each matrix column as its own series
pie(x)            Pie chart
curve(expr,from,to) Plot a function/expression in x (add=TRUE to overlay)
lines(x,y)        Add a line (data coords; errors if no plot is open)
points(x,y)       Add points (data coords; col/pch/cex)
abline(a,b)       Add reference line: intercept/slope, abline(h=)/abline(v=),
                  or abline(lm(y~x)) to draw a fitted regression line
text(x,y,labels)  Add text labels at data coordinates (pos=)
title(main=,...)  Add main/sub/xlab/ylab to the current plot
axis(side,at=)    Draw an axis (side 1=bottom,2=left,3=top,4=right)
rect(x1,y1,x2,y2) Draw rectangle(s) in data coordinates
legend(...)       Add legend
pdf(file,w,h)     Open a vector-PDF file device (dev.off() writes it)
png(file,w,h)     Open a raster-PNG file device
svg(file,w,h)     Open a vector-SVG file device
par(...)          Get or set graphical parameters. Supported:
                  col, bg, fg (colours); cex (scale); lwd, lty (lines);
                  pch (point symbol); las (axis-label rotation: 0/1/2/3);
                  mar, oma (margins); mfrow/mfcol (multi-panel grid); new.
                  par() — return all current params as a named list
                  par("col") — return single param
                  par(col="red", lwd=2, las=2) — set; returns previous values
                  par(mfrow=c(2,2)) — enable 2x2 multi-panel layout
                  par(mfcol=c(2,3)) — column-major multi-panel layout
                  oldpar <- par(cex=1.5); par(oldpar)  # save/restore
dev.off()         Close current graphics device (reset to defaults)
dev.view()        Start the built-in HTTP plot viewer and open browser
                  at http://127.0.0.1:8765/ . Two-pane layout: live
                  current plot at top, session gallery below. Click any
                  gallery thumbnail to pin the top pane to that file.
save_plot(path)   Explicitly flush the current device's SVG to a file
```

Supported `par()` parameters: `mfrow`, `mfcol`, `mar`, `oma`, `cex`,
`cex.axis`, `cex.lab`, `cex.main`, `col`, `bg`, `fg`, `lty`, `lwd`,
`pch`, `las`, `new`. Defaults match CRAN R 4.5.x.

## Model Functions (11)
```
lmer(y ~ x + (1|g), data=)   Random-intercept mixed model by profiled REML
plssem(model, data=, seed=) / csem(model, data=)   PLS path model (PLS-SEM, cSEM defaults) with bootstrap
mmap.lm(y, x, ...)  lm over memory-mapped columns (out-of-core; normal equations)
predict(model,newdata)  Predict from model
residuals(model)        Residuals
fitted(model)           Fitted values
coef(model)             Coefficients
deviance(model)         Residual deviance (lm/glm)
summary(model)          Model summary (auto-dispatch)
plot(model)             Model diagnostic plot (auto-dispatch)
```

## Performance & Parallelism (11)
```
explain(f)          Report whether closure f JIT-compiles (and to what), or
                    exactly which construct keeps it on the interpreter
explain(x)          For data: size, architecture (SIMD/cores), and whether
                    operations on x will run serial or parallel
mclapply(x, FUN)    Run FUN over x across CPU cores (isolated workers,
                    per-worker reproducible RNG); par.lapply is an alias
par.lapply(x, FUN)  Alias of mclapply
par.sapply(x, FUN)  Parallel sapply — simplifies to vector/matrix
mem.budget("8G")    Set (or with no argument, read) the memory budget in bytes; "512M", "8G" or a number
mem.status()        list(live.bytes, budget.bytes, spills, over.budget)
mem.spill(x, dir=)  Write a numeric vector to a memory-mapped file and return the mapped column
mem.restore(m)      Bring a spilled column back into RAM
gpu.enable(TRUE)    Turn the GPU path on or off; returns the current state
gpu.status()        Adapter name, whether enabled, whether available
gpu.map("relu", x)  Run an elementwise op on the GPU: identity, relu, sigmoid, tanh, exp, scale
```

## System (26)
```
library(pkg)        Load package
detach(pkg)         Unload package
require(pkg)        Try to load package
search()            Search path
help(topic)         Help on topic (also ?topic, ??topic)
version()           Ardon-R2 version info
getwd()             Working directory
setwd(path)         Change directory
Sys.time()          Current time
Sys.getenv(var)     Environment variable
Sys.sleep(n)        Pause n seconds
system.time(expr)   Time an expression
strict() / lenient()   Error mode: strict stops on the first error (default); lenient reports and continues
clear() / cls() / clr()   Clear the console
.libPaths()         Package library search directories
.Internal(name, ...)   Call an engine primitive directly (crossprod, matmul, solve, ...)
install.packages(name)   Install an R2 package by name
install.from.github("user/repo")   Install a package from a GitHub repository
install.from.zip(path)   Install from a .zip archive
install.from.dir(path)   Install from a directory of .r2 source
installed.packages()   Data frame of installed packages
uninstall(name)     Remove an installed package
readline(prompt)    Block until stdin line is entered; returns character.
                    Used for interactive prompts in scripts:
                      name <- readline("Your name: ")
                      ans  <- readline("Save as [default.svg]: ")
                      invisible(readline("Press Enter to continue..."))
```

## LLM training & serving (7)
```
llm.new(dim=128, layers=4, heads=4, kv.heads=heads, ffn=3*dim, vocab=8000, ctx=64, lr=0.003, seed=42)
                    Create a transformer model; returns an integer handle
llm.train(m, text, steps=50, batch=8, seq=16, report=0)
                    Train handle m on a text (string or file path); report= prints every n steps
llm.generate(m, prompt, n=32, temperature=0, top.k=0, seed=1)
                    Generate n tokens from a prompt (temperature 0 = greedy)
llm.save(m, dir)    Write weights + tokenizer (safetensors + HuggingFace tokenizer.json)
llm.load(dir)       Load a saved model for serving (no optimizer state); returns a handle
llm.info(m)         Configuration and parameter count of a handle
llm.free(m)         Release a handle's memory
```

## Language / Metaprogramming (10)
```
quote(expr)        Capture an expression unevaluated (a language object)
eval(expr)         Evaluate a language object (e.g. eval(parse(text=...)))
parse(text=)       Parse source text into language object(s)
deparse(expr)      Turn a language object back into source text
call(name, ...)    Build an unevaluated call: call("sum",1,2) → sum(1, 2)
as.call(list)      Turn a list (function + args) into a call
body(f)            Body of a user-defined function (a language object)
formals(f)         Formal arguments as a named list (defaults or NULL)
args(f)            Function signature (same formals, NULL body)
substitute(expr)   Replace a function's params with the caller's expressions
match.call()       Current call with args matched to formal names
sys.call()         Current call exactly as written
bquote(expr)       Quote expr, splicing in any .(x) evaluated inline
```
Note: arguments are evaluated eagerly (no lazy promises), so `substitute`
captures the caller's expression for labeling/deparse but the argument must
still be evaluable. See docs/PHASE_L_LANGUAGE_OBJECTS.md.

## Operators (21)
```
<-  =           Assignment
+  -  *  /      Arithmetic
^  %%  %/%      Power, modulo, integer division
%*%             Matrix multiply
%in%            Membership test (x %in% y → logical); any %name% infix works
~               Formula
|>              Pipe
::              Package access
$               Column/field access
==  !=          Equality
<  >  <=  >=   Comparison
&  |  &&  ||   Logical
!               Negation
```
