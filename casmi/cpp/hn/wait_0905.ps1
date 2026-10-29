# wait_0905.ps1: sleep until 09:05 local time (Kaggle's daily submission limit resets at 00:00 UTC = 09:00 JST)
$now = Get-Date
$t = $now.Date.AddHours(9).AddMinutes(5)
if ($now -ge $t) { $t = $t.AddDays(1) }
Write-Output "waiting until $t"
while ((Get-Date) -lt $t) { Start-Sleep -Seconds 300 }
Write-Output "wait over $(Get-Date)"
