<#
Run degli esperimenti exp01 (cartella scripts/) in ordine:
 1. exp01_synth.py
 2. exp01_hopfield_panel.py
 3. exp01_mnist.py
 4. exp01_sweep_synth.py

Uso:
  powershell -ExecutionPolicy Bypass -File .\run_exp01_all.ps1 [-PythonExe python] [-StopOnError]

Parametri:
  -PythonExe   Path o nome eseguibile Python (default: 'python')
  -StopOnError Se presente, lo script si ferma al primo fallimento

Lo script crea una cartella logs/ per stdout/stderr di ciascun step.
#>
param(
    [string]$PythonExe = "python",
    [switch]$StopOnError
)

$ErrorActionPreference = 'Stop'

$steps = @(
    @{ name = 'exp01_synth';          script = 'scripts/exp01_synth.py' },
    @{ name = 'exp01_hopfield_panel'; script = 'scripts/exp01_hopfield_panel.py' },
    @{ name = 'exp01_mnist';          script = 'scripts/exp01_mnist.py' },
    @{ name = 'exp01_sweep_synth';    script = 'scripts/exp01_sweep_synth.py' }
)

# Directory logs/logs_exp01/
$logsRoot = Join-Path -Path (Get-Location) -ChildPath 'logs'
if (-not (Test-Path $logsRoot)) { New-Item -ItemType Directory -Path $logsRoot | Out-Null }
$logDir = Join-Path -Path $logsRoot -ChildPath 'logs_exp01'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

function Run-Step {
    param(
        [string]$Name,
        [string]$ScriptPath
    )
    Write-Host "==================== $Name ====================" -ForegroundColor Cyan
    $logOut = Join-Path $logDir "$Name.out.log"
    $logErr = Join-Path $logDir "$Name.err.log"
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[$ts] Avvio: $ScriptPath"

    # Ensure PYTHONPATH for the child process
    $env:PYTHONPATH = (Get-Location).ProviderPath

    # Use Start-Transcript to record everything printed to the console for this step.
    # Start-Transcript streams output live to console and saves to the transcript file.
    Write-Host "[run] Eseguo: $PythonExe -u $ScriptPath (trascrivendo in $logOut)" -ForegroundColor Yellow
    try {
        Start-Transcript -Path $logOut -Force | Out-Null
    } catch {
        Write-Host "[warn] Start-Transcript non disponibile o fallito; verrà eseguito senza log trascritto." -ForegroundColor Yellow
    }

    # Run the script (unbuffered) so output appears promptly
    $previousEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $PythonExe -u $ScriptPath
        $exit = $LASTEXITCODE
    } catch {
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousEAP
    }

    # Stop transcript if it was started
    try {
        Stop-Transcript | Out-Null
    } catch {
        # non critico se Stop-Transcript fallisce
    }

    # If transcript didn't run, ensure there's at least an (empty) log file
    if (-not (Test-Path $logOut)) { New-Item -Path $logOut -ItemType File | Out-Null }
    # Duplicate combined log to err log for compatibility
    try { Copy-Item -Path $logOut -Destination $logErr -Force } catch { }

    if ($exit -ne 0) {
        Write-Host "[ERRORE] Step '$Name' terminato con codice $exit." -ForegroundColor Red
        # show tail of log for quick debugging
        if (Test-Path $logOut) {
            Write-Host "---- Ultime righe log ($logOut) ----" -ForegroundColor DarkRed
            Get-Content $logOut -Tail 30 | ForEach-Object { Write-Host $_ }
            Write-Host "-------------------------------------" -ForegroundColor DarkRed
        }
        if ($StopOnError) { throw "Interrotto per errore nello step $Name" }
    } else {
        Write-Host "[OK] $Name completato (ExitCode=0). Log: $logOut" -ForegroundColor Green
    }
}

foreach ($s in $steps) {
    Run-Step -Name $s.name -ScriptPath $s.script
}

Write-Host "Tutti gli step exp01 completati." -ForegroundColor Green
