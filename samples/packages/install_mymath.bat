@echo off
REM Install R2 sample addon package: mymath
REM This copies the package to your R2 library path

set R2LIB=%USERPROFILE%\.r2\library

echo Installing mymath package to %R2LIB%\mymath ...

if not exist "%R2LIB%" mkdir "%R2LIB%"
if not exist "%R2LIB%\mymath" mkdir "%R2LIB%\mymath"
if not exist "%R2LIB%\mymath\R2" mkdir "%R2LIB%\mymath\R2"

copy /Y "%~dp0mymath\R2\functions.r" "%R2LIB%\mymath\R2\functions.r"

echo Done! Start R2 and type: library(mymath)
echo.
echo Available functions:
echo   factorial(5)       = 120
echo   fibonacci(10)      = 55
echo   gcd(12, 8)         = 4
echo   lcm(4, 6)          = 12
echo   sigmoid(0)         = 0.5
echo   normalize(c(1,5,3))
echo   rmse(actual, predicted)
echo   mae(actual, predicted)
pause
