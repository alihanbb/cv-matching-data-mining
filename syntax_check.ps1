$ErrorActionPreference = "Continue"

$files = @(
    "main.py",
    "src/models/learned_fusion.py",
    "src/models/cross_encoder_rerank.py",
    "src/extraction/skill_extractor.py",
    "api/main.py",
    "api/routers/cv.py",
    "api/routers/job.py",
    "api/routers/ranking.py",
    "tests/test_api_routers.py"
)

$hasErrors = $false

foreach ($file in $files) {
    Write-Host "Checking: $file"
    python -m py_compile $file 2>&1
    if ($LASTEXITCODE -ne 0) {
        $hasErrors = $true
    }
}

if (-not $hasErrors) {
    Write-Host "All OK"
} else {
    Write-Host "Some files have syntax errors"
}
