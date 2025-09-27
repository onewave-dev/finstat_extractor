@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "ICON_OPTION="
if exist "%SCRIPT_DIR%assets\app.ico" (
    set "ICON_OPTION=--icon \"%SCRIPT_DIR%assets\app.ico\""
)

pyinstaller ^
    --name app ^
    --onefile "%SCRIPT_DIR%app.py" ^
    --add-data "%SCRIPT_DIR%config.yaml;." ^
    --version-file "%SCRIPT_DIR%metadata\version_info.txt" ^
    --clean ^
    --noconfirm %ICON_OPTION%

endlocal