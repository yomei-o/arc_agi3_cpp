Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'e2e_run|phys_rank|e2e36|phys_v36pc' } |
  Select-Object ProcessId, ParentProcessId, CreationDate, Name, @{n='cmd';e={$_.CommandLine.Substring(0, [Math]::Min(120, $_.CommandLine.Length))}} | Format-Table -AutoSize -Wrap
