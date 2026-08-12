@echo off
rem tw - the pretty twill CLI, written in twill and run on the bootstrap binary.
rem %~dp0 is this script's own directory, so the launcher works from anywhere.
"%~dp0twill.exe" run "%~dp0src\cli\main.tw" %*
