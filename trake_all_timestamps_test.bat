@echo off
setlocal enabledelayedexpansion
set "base_path=%cd%"
set "image_path=%base_path%\test\images"
for /d %%i in ("%image_path%\*") do (
    set "folder_path=%%i"
    echo "!folder_path!"
    python tools\mc_demo_yolov7.py --source "!folder_path!" --name "%%~nxi" --project runs/test
)