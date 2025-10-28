@echo off
setlocal enabledelayedexpansion

:: Initialize total line count
set total=0

:: Loop through all .lua files recursively
for /r %%f in (*.lua) do (
    set /a lines=0
    for /f "usebackq delims=" %%l in ("%%f") do (
        set /a lines+=1
    )
    echo %%f : !lines! lines
    set /a total+=lines
)

echo.
echo Total lines in all Lua files: %total%
pause
