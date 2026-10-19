# launch.ps1 <name> <bat file>  -- start a long job on the build machine that survives the ssh session.
#
# Windows OpenSSH kills the whole process tree of a session when it ends, so
# `start /B`, Start-Process etc. die with it. A process created through WMI
# (Win32_Process.Create) is a child of the WMI provider host instead, so it
# keeps running after the ssh session closes. This is a one-shot launch of an
# ordinary user process (no scheduled task, no service, nothing persistent).
#
# Each job writes C:\prog\casmi\night\<name>.log, and <name>.pid holds the cmd PID
# so the job can be checked or stopped by PID later:
#   powershell -c "Get-Process -Id (Get-Content C:\prog\casmi\night\<name>.pid)"
#   taskkill /pid <pid> /t /f        (only this job's tree)
param([string]$Name, [string]$Bat)
$dir = "C:\prog\casmi\night"
New-Item -ItemType Directory -Force $dir | Out-Null
$log = Join-Path $dir "$Name.log"
# cmd /c strips the first and last quote when there are more than two, so the
# whole command gets one extra pair: cmd /c ""bat" > "log" 2>&1"
$cmd = "cmd /c `"`"$Bat`" > `"$log`" 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = $dir}
if ($r.ReturnValue -ne 0) { Write-Output "launch failed: $($r.ReturnValue)"; exit 1 }
Set-Content -Path (Join-Path $dir "$Name.pid") -Value $r.ProcessId
Write-Output "started $Name pid $($r.ProcessId) log $log"
