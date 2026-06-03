# Ardon-R2 Function Reference — 216 Built-in Functions

## Core (25)
```
c(...)              Create vector: c(1,2,3)
length(x)           Length of vector
print(x)            Print value
cat(...)            Print without newline
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
is.na(x)            Test for NA
is.numeric(x)       Test if numeric
is.character(x)     Test if character
is.logical(x)       Test if logical
is.null(x)          Test if NULL
is.data.frame(x)    Test if data.frame
is.factor(x)        Test if factor
is.matrix(x)        Test if matrix
TRUE/FALSE/T/F      Logical constants
NA                  Missing value
NULL                Null value
```

## Math (18)
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
```

## Statistics (24)
```
mean(x)         Arithmetic mean
sd(x)           Standard deviation
var(x)          Variance
cor(x,y)        Correlation
median(x)       Median
quantile(x,p)   Quantile at probability p
lm(y~x,data)    Linear regression
glm(y~x,data,family) Generalized linear model
aov(y~x,data)   One-way analysis of variance
                Repeated measures (Phase R.S.1):
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
                Phase R.S.1 extensions (not supported by R itself):
                  t.test(y ~ group + Error(subject), paired=TRUE, data=df)
                  t.test(y ~ Error(subject), paired=TRUE, data=df)
                    (pairs each subject's 2 obs by row order)
chisq.test(x)   Chi-squared test
hotelling.test  Multivariate Hotelling's T² (Phase R.S.2)
                One-sample:    hotelling.test(X)
                With null mu:  hotelling.test(X, mu=c(0,0,0))
                Two-sample:    hotelling.test(X, Y)
                Paired/RM:     hotelling.test(X, Y, paired=TRUE)
                X and Y are n×p matrices of multivariate observations.
                Returns T², F, df, p-value.
manova(formula,data) Multivariate ANOVA (Phase R.S.2)
                LHS is a multivariate response (use cbind):
                  manova(cbind(y1, y2, y3) ~ group, data=df)
                Reports four classical statistics:
                  Wilks' Lambda (with Rao F-approximation + p-value)
                  Pillai's trace
                  Hotelling-Lawley trace
                  Roy's largest root
                Returns TypeInstance with all four + eigenvalues vector.
confint(model)  Confidence intervals
rnorm(n)        Random normal
runif(n)        Random uniform
rbinom(n,size,prob) Random binomial
rpois(n,lambda) Random Poisson
dnorm(x)        Normal density
pnorm(x)        Normal CDF
qnorm(p)        Normal quantile
set.seed(n)     Set random seed
sample(x,n)     Random sample
```

## Machine Learning (12)
```
rpart(y~.,data)         Decision tree (CART)
rf(y~.,data,ntrees)     Random forest
gbm(y~.,data,ntrees)    Gradient boosted trees
kmeans(x,centers)       K-means clustering
knn(train,test,labels,k) K-nearest neighbors
naive.bayes(x,y)        Gaussian naive Bayes
prcomp(x)               Principal component analysis
svd(x)                  Singular value decomposition
eigen(x)                Eigenvalue decomposition
scale(x)                Center and scale
cv(x,y,model,k)         K-fold cross-validation
confusion.matrix(pred,actual) Confusion matrix + F1
```

## Data Handling (30)
```
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
```

## String Functions (16)
```
paste(...,sep)      Concatenate with separator
paste0(...)         Concatenate without separator
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
```

## Apply Family (6)
```
sapply(x,fun)       Apply and simplify
lapply(x,fun)       Apply and return list
apply(x,margin,fun) Apply over matrix margins
tapply(x,idx,fun)   Apply by groups
aggregate(x,by,fun) Aggregate by groups (also: aggregate(y ~ g, data=df, FUN))
do.call(fun,args)   Call function with arg list
```

## I/O (10)
```
read.csv(file)      Read CSV file
write.csv(x,file)   Write CSV file
read.table(file)    Read delimited file
write.table(x,file) Write delimited file
read.delim(file)    Read tab-delimited
source(file)        Run R2 script
save(file)          Save session
load(file)          Load session
file.exists(path)   Check if file exists
list.files(path)    List directory
```

## Graphics (12)
```
plot(x,y)         Scatter plot (SVG, draws into in-memory device)
hist(x)           Histogram
boxplot(x)        Box-and-whisker
barplot(x)        Bar chart
lines(x,y)        Add lines to plot (errors if no plot is open)
points(x,y)       Add points
abline(a,b)       Add reference line (also abline(h=) / abline(v=))
legend(...)       Add legend
par(...)          Get or set graphical parameters
                  par() — return all current params as a named list
                  par("col") — return single param
                  par(col="red", lwd=2) — set; returns previous values
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

## Model Functions (6)
```
predict(model,newdata)  Predict from model
residuals(model)        Residuals
fitted(model)           Fitted values
coef(model)             Coefficients
summary(model)          Model summary (auto-dispatch)
plot(model)             Model diagnostic plot (auto-dispatch)
```

## System (13)
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
readline(prompt)    Block until stdin line is entered; returns character.
                    Used for interactive prompts in scripts:
                      name <- readline("Your name: ")
                      ans  <- readline("Save as [default.svg]: ")
                      invisible(readline("Press Enter to continue..."))
```

## Operators (20)
```
<-  =           Assignment
+  -  *  /      Arithmetic
^  %%  %/%      Power, modulo, integer division
%*%             Matrix multiply
~               Formula
|>              Pipe
::              Package access
$               Column/field access
==  !=          Equality
<  >  <=  >=   Comparison
&  |  &&  ||   Logical
!               Negation
```
