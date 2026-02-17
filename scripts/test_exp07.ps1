# Quick test script for Exp-07 with entropy-based adaptive damping
# Run this to verify the implementation works correctly

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "EXP-07 QUICK TEST - Entropy-Based Adaptive Damping" -ForegroundColor Yellow
Write-Host ("=" * 80) -ForegroundColor Cyan

Write-Host "`n[1/4] Testing baseline strategy..." -ForegroundColor Green
python scripts\exp07_novelty_emergence.py `
    --strategy baseline `
    --n-seeds 6 `
    --n-batch 24 `
    --K-old 3 `
    --K-new 3 `
    --N 400 `
    --no-progress `
    --out-dir out_07\baseline

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Baseline test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[2/4] Testing EMA damping..." -ForegroundColor Green
python scripts\exp07_novelty_emergence.py `
    --strategy ema `
    --alpha-ema 0.3 `
    --n-seeds 6 `
    --n-batch 24 `
    --K-old 3 `
    --K-new 3 `
    --N 400 `
    --no-progress `
    --out-dir out_07\ema

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: EMA test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[3/4] Testing rate_limit damping..." -ForegroundColor Green
python scripts\exp07_novelty_emergence.py `
    --strategy rate_limit `
    --max-delta-w 0.15 `
    --n-seeds 6 `
    --n-batch 24 `
    --K-old 3 `
    --K-new 3 `
    --N 400 `
    --no-progress `
    --out-dir out_07\rlimit

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Rate limit test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[4/4] Testing adaptive_ema (full test with comparison)..." -ForegroundColor Green
python scripts\exp07_novelty_emergence.py `
    --strategy baseline adaptive_ema `
    --alpha-min 0.1 `
    --alpha-max-adapt 0.5 `
    --n-seeds 6 `
    --n-batch 24 `
    --K-old 3 `
    --K-new 3 `
    --N 400 `
    --no-progress `
    --out-dir out_07\adaema

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Adaptive EMA test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n" -NoNewline
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "ALL TESTS PASSED!" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan

Write-Host "`nYou can now run full experiments with:" -ForegroundColor Yellow
Write-Host "  python scripts\exp07_novelty_emergence.py --strategy baseline ema adaptive_ema" -ForegroundColor White
Write-Host "`nFor help and all options:" -ForegroundColor Yellow
Write-Host "  python scripts\exp07_novelty_emergence.py --help" -ForegroundColor White
