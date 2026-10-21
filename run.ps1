# x86 kit runner for Windows. Everything stays inside this folder: a Python venv
# (or, without any Python on the PC, the python.org embeddable zip unpacked here),
# the pip packages, the compiled helpers and the results. No installer, no PATH,
# registry or driver change. Delete the folder afterwards.
#
#   powershell -ExecutionPolicy Bypass -File run.ps1 [-Seeds 20]
#
# Run it from "Developer PowerShell for VS" if Visual Studio is installed: then the
# game's host build (hostsim) and the portable kernel are compiled too. Without a
# compiler the replay part still runs; that is the main result.
param([int]$Seeds = 20)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "run.log"
Start-Transcript -Path $log -Append | Out-Null
Write-Host "== x86 kit in $PSScriptRoot"

# ---------------------------------------------------------------- Python
function Test-Py($exe, $args) {
    try {
        $v = & $exe @args -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(9|1[0-3])$') { return $v }
    } catch {}
    return $null
}
$py = $null; $pyArgs = @()
foreach ($c in @(@("py", @("-3.12")), @("py", @("-3.11")), @("py", @("-3.13")), @("py", @("-3.10")), @("py", @("-3")), @("python", @()))) {
    $v = Test-Py $c[0] $c[1]
    if ($v) { $py = $c[0]; $pyArgs = $c[1]; Write-Host "Python $v found ($py $pyArgs)"; break }
}
if ($py) {
    if (-not (Test-Path ".venv\Scripts\python.exe")) { & $py @pyArgs -m venv .venv }
    $venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
} else {
    Write-Host "No Python 3.9-3.13 on this PC: unpacking the python.org embeddable zip into this folder (no install)."
    if (-not (Test-Path "python-embed\python.exe")) {
        Invoke-WebRequest "https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip" -OutFile python-embed.zip
        Expand-Archive python-embed.zip -DestinationPath python-embed -Force
        $pth = Get-ChildItem "python-embed\python3*._pth" | Select-Object -First 1
        (Get-Content $pth.FullName) -replace '^#import site', 'import site' | Set-Content $pth.FullName
        Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile get-pip.py
        & "python-embed\python.exe" get-pip.py --no-warn-script-location
    }
    $venvPy = Join-Path $PSScriptRoot "python-embed\python.exe"
}
& $venvPy -m pip install --quiet --upgrade pip
& $venvPy -m pip install --quiet numpy "tensorflow==2.20.0"
& $venvPy -c "import tensorflow as tf, numpy; print('TensorFlow', tf.__version__, 'numpy', numpy.__version__)"

# ---------------------------------------------------------------- compiler (optional)
$cl = Get-Command cl -ErrorAction SilentlyContinue
if (-not $cl) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vs) {
            $bat = Join-Path $vs "Common7\Tools\VsDevCmd.bat"
            cmd /c "`"$bat`" -arch=amd64 -no_logo && set" | ForEach-Object {
                if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
            }
            $cl = Get-Command cl -ErrorAction SilentlyContinue
        }
    }
}
New-Item -ItemType Directory -Force build | Out-Null
if ($cl) {
    Write-Host "MSVC found: building the host game (three float settings) and the portable kernel"
    $src = "sim\hostsim.cpp sim\src\game.cpp sim\src\trace.cpp"
    $common = "/nologo /std:c++20 /O2 /EHsc /D_CRT_SECURE_NO_WARNINGS /I sim\src /I sim\stubs /Fobuild\"
    foreach ($v in @(@("hostsim_strict", "/fp:strict"), @("hostsim_precise", "/fp:precise"), @("hostsim_contract", "/arch:AVX2 /fp:contract"))) {
        try {
            $cmd = "cl $common $($v[1]) $src /Fe:$($v[0]).exe"
            Write-Host "  $cmd"
            cmd /c $cmd | Out-Null
            if (-not (Test-Path "$($v[0]).exe")) { Write-Host "  $($v[0]): build failed (skipped)" }
        } catch { Write-Host "  $($v[0]): build failed (skipped)" }
    }
    try {
        cmd /c "cl /nologo /O2 /fp:strict /TC /D_CRT_SECURE_NO_WARNINGS /I portable /Fobuild\ portable\pp_cli.c portable\policy_portable.c /Fe:pp_cli.exe" | Out-Null
    } catch { Write-Host "  pp_cli: build failed (skipped)" }
} else {
    $cxx = Get-Command clang++ -ErrorAction SilentlyContinue
    if (-not $cxx) { $cxx = Get-Command g++ -ErrorAction SilentlyContinue }
    if ($cxx) {
        Write-Host "$($cxx.Name) found: building the host game and the portable kernel"
        $cc = if ($cxx.Name -like "clang*") { "clang" } else { "gcc" }
        try {
            & $cxx.Name -std=c++20 -O2 -ffp-contract=off -Isim/stubs -Isim/src sim/hostsim.cpp sim/src/game.cpp sim/src/trace.cpp -o hostsim_strict.exe
            & $cxx.Name -std=c++20 -O2 -Isim/stubs -Isim/src sim/hostsim.cpp sim/src/game.cpp sim/src/trace.cpp -o hostsim_contract.exe
            & $cc -O2 -ffp-contract=off -fno-fast-math -std=c99 -o pp_cli.exe portable/pp_cli.c portable/policy_portable.c
        } catch { Write-Host "  build failed (skipped)" }
    } else {
        Write-Host "No C/C++ compiler on PATH: the replay part runs, the game re-simulation and seeds are skipped."
    }
}

# ---------------------------------------------------------------- run
$name = $env:COMPUTERNAME.ToLower()
& $venvPy x86_replay.py --out "results\$name" --seeds $Seeds
Write-Host ""
Write-Host "== done. Results: results\$name\RESULTS.md (and results.json, the replay .bin files)"
Write-Host "Next: git add results; git commit -m `"x86 results $name`"; git push"
Stop-Transcript | Out-Null
