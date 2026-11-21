@echo off
rem CAVA demo: backend API + public tunnel, each in its own window (closing this window does not stop them).
rem The tunnel URL changes on every start: it is printed here and copied to the clipboard.
chcp 65001 >nul
cd /d "%~dp0"
set "LOG=%TEMP%\cava-tunnel.log"
if exist "%LOG%" del "%LOG%"

start "CAVA 백엔드 (이 창을 닫으면 꺼짐)" cmd /k ".venv\Scripts\python.exe -m main serve"
start "CAVA 터널 (이 창을 닫으면 꺼짐)" cmd /k ""C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --no-autoupdate --url http://127.0.0.1:8000 --logfile "%LOG%""

echo 터널 주소를 기다리는 중...
powershell -NoProfile -Command "$u=$null; for($i=0; $i -lt 60 -and -not $u; $i++){ Start-Sleep 1; if(Test-Path $env:LOG){ $m = Select-String -Path $env:LOG -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1; if($m){ $u = $m.Matches[0].Value } } }; if($u){ Set-Clipboard $u; Write-Host ''; Write-Host ('백엔드 주소: ' + $u); Write-Host '클립보드에 복사됨. 앱의 설정 > 백엔드 주소에 붙여넣고 저장하세요.' } else { Write-Host '주소를 찾지 못함. 터널 창을 확인하세요.' }"
echo.
pause
