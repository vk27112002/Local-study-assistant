$ErrorActionPreference = "Stop"

$appPort = 8501
$listeners = @(
    Get-NetTCPConnection -State Listen -LocalPort $appPort -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
)

foreach ($ownerPid in $listeners) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerPid"
    if ($owner.CommandLine -notmatch "streamlit") {
        throw "Port $appPort is occupied by a non-Streamlit process (PID $ownerPid)."
    }
    $streamlitPids = @($ownerPid)
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($owner.ParentProcessId)"
    if ($parent.CommandLine -match "streamlit.*app.py") {
        $streamlitPids += $parent.ProcessId
    }
    Stop-Process -Id $streamlitPids -Force -ErrorAction SilentlyContinue
}

for ($attempt = 0; $attempt -lt 20; $attempt++) {
    if (-not (Get-NetTCPConnection -State Listen -LocalPort $appPort -ErrorAction SilentlyContinue)) {
        break
    }
    Start-Sleep -Milliseconds 250
}
if (Get-NetTCPConnection -State Listen -LocalPort $appPort -ErrorAction SilentlyContinue) {
    throw "Port $appPort did not become available after stopping Streamlit."
}
$projectPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw "The rag_project virtual environment is missing."
}

Set-Location -LiteralPath $PSScriptRoot
& $projectPython -m streamlit run app.py --server.port $appPort
