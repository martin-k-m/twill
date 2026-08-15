# Regenerates every number in docs/BENCHMARKS.md.
#
# Run from the repository root:
#
#     powershell -File bench/run_all.ps1 -Python <path-to-python-with-torch>
#
# Results land in bench/results/ as both the raw console output and JSON. The
# PyTorch step is skipped, loudly, if the interpreter given has no torch: a
# missing comparison is reported rather than silently leaving twill's numbers
# standing on their own.

param(
    [string]$Python = "python",
    [int]$Runs = 30,
    [int]$Warmup = 5,
    [string]$Procs = "1,2,4,8,16",
    [string]$OutDir = "bench/results"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "== environment =="
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$os = Get-CimInstance Win32_OperatingSystem
$env_lines = @(
    "cpu: $($cpu.Name)"
    "cores: $($cpu.NumberOfCores) physical, $($cpu.NumberOfLogicalProcessors) logical"
    "base_clock_mhz: $($cpu.MaxClockSpeed)"
    "ram_gb: $([math]::Round($os.TotalVisibleMemorySize / 1MB, 2))"
    "os: $($os.Caption) build $($os.BuildNumber)"
    "go: $(go version)"
)
foreach ($g in Get-CimInstance Win32_VideoController) { $env_lines += "gpu_present: $($g.Name)" }
$env_lines | Tee-Object -FilePath "$OutDir/environment.txt"

Write-Host "`n== twill workloads (GOMAXPROCS sweep) =="
go run ./bench/cmd/twillbench -runs $Runs -warmup $Warmup -procs $Procs `
    -out "$OutDir/twill.json" | Tee-Object -FilePath "$OutDir/twill.txt"

Write-Host "`n== front end: lex, parse, shape check =="
go run ./bench/cmd/checkbench -runs $Runs -warmup $Warmup `
    -out "$OutDir/checker.json" | Tee-Object -FilePath "$OutDir/checker.txt"

Write-Host "`n== gradient check over the full operator set =="
go test ./internal/tensor/ -run 'TestGradientCheck' -v -count=1 |
    Tee-Object -FilePath "$OutDir/gradcheck.txt"

Write-Host "`n== PyTorch comparison =="
$hasTorch = $false
try {
    & $Python -c "import torch" 2>$null
    if ($LASTEXITCODE -eq 0) { $hasTorch = $true }
} catch { $hasTorch = $false }

if ($hasTorch) {
    & $Python bench/torch_bench.py --threads $Procs --runs $Runs --warmup $Warmup `
        --out "$OutDir/torch.json" | Tee-Object -FilePath "$OutDir/torch.txt"
} else {
    $msg = "SKIPPED: '$Python' cannot import torch. The twill numbers above stand alone; " +
           "the comparison in docs/BENCHMARKS.md was not regenerated."
    Write-Warning $msg
    $msg | Out-File "$OutDir/torch.txt"
}

Write-Host "`n== profile: where the Monte Carlo pricer's time goes =="
go test ./bench/profile/ -run "TestProfileMonteCarlo" -count=1 `
    -cpuprofile "$OutDir/mc.prof" -o "$OutDir/interp.test.exe" |
    Tee-Object -FilePath "$OutDir/profile.txt"
if (Test-Path "$OutDir/mc.prof") {
    go tool pprof -top -nodecount=20 "$OutDir/interp.test.exe" "$OutDir/mc.prof" |
        Tee-Object -Append -FilePath "$OutDir/profile.txt"
}

Write-Host "`nDone. Results in $OutDir/"
