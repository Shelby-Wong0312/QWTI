# ============================================================
# WTI Auto Deploy Script
# ============================================================

# Settings - EDIT HERE
$KeyPath       = "$env:USERPROFILE\.ssh\data-ec2-key.pem"
$LocalRepoPath = "C:\Users\niuji\Documents\Data\capital_wti_downloader"
$EnvFilePath   = "C:\Users\niuji\Documents\Data\capital_wti_downloader\.env"
$EC2_IP        = "3.236.235.113"
$EC2_User      = "ec2-user"
$RemoteDir     = "wti"

# ============================================================
Write-Host "========================================"
Write-Host "  WTI Auto Deploy Script - Starting"
Write-Host "========================================"

# Check SSH
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] SSH not found" -ForegroundColor Red
    exit 1
}

# Check .pem file
if (-not (Test-Path $KeyPath)) {
    Write-Host "[ERROR] SSH Key not found: $KeyPath" -ForegroundColor Red
    exit 1
}

# Check local repo
if (-not (Test-Path $LocalRepoPath)) {
    Write-Host "[ERROR] Project folder not found: $LocalRepoPath" -ForegroundColor Red
    exit 1
}

# Check .env file
if (-not (Test-Path $EnvFilePath)) {
    Write-Host "[ERROR] .env file not found: $EnvFilePath" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] All files verified" -ForegroundColor Green

$RemoteHost = "${EC2_User}@${EC2_IP}"

# ============================================================
# Step 1: Clean remote directory
# ============================================================
Write-Host ""
Write-Host "[Step 1/5] Cleaning remote directory..." -ForegroundColor Yellow

ssh -i $KeyPath -o StrictHostKeyChecking=no $RemoteHost "rm -rf ~/wti; mkdir -p ~/wti/logs"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to clean remote directory" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Remote directory cleaned" -ForegroundColor Green

# ============================================================
# Step 2: Upload project files
# ============================================================
Write-Host ""
Write-Host "[Step 2/5] Uploading project files..." -ForegroundColor Yellow

$scpDest = "${RemoteHost}:~/wti/"
scp -i $KeyPath -o StrictHostKeyChecking=no -r "${LocalRepoPath}\*" $scpDest

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to upload project files" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Project files uploaded" -ForegroundColor Green

# ============================================================
# Step 3: Upload .env file
# ============================================================
Write-Host ""
Write-Host "[Step 3/5] Uploading .env file..." -ForegroundColor Yellow

$envDest = "${RemoteHost}:~/wti/.env"
scp -i $KeyPath -o StrictHostKeyChecking=no $EnvFilePath $envDest

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to upload .env file" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] .env file uploaded" -ForegroundColor Green

# ============================================================
# Step 4: Setup remote environment
# ============================================================
Write-Host ""
Write-Host "[Step 4/5] Setting up remote environment..." -ForegroundColor Yellow

# Create swap if not exists
ssh -i $KeyPath -o StrictHostKeyChecking=no $RemoteHost "if [ ! -f /swapfile ]; then sudo fallocate -l 4G /swapfile; sudo chmod 600 /swapfile; sudo mkswap /swapfile; sudo swapon /swapfile; echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab; fi"

# Fix .env permissions and install requirements
ssh -i $KeyPath -o StrictHostKeyChecking=no $RemoteHost "cd ~/wti; chmod 600 .env; sed -i 's/\r$//' .env; if [ -f requirements.txt ]; then pip3 install -r requirements.txt --user --quiet; fi; mkdir -p logs"

Write-Host "[OK] Remote environment setup complete" -ForegroundColor Green

# ============================================================
# Step 5: Setup Cron job
# ============================================================
Write-Host ""
Write-Host "[Step 5/5] Setting up Cron job..." -ForegroundColor Yellow

ssh -i $KeyPath -o StrictHostKeyChecking=no $RemoteHost "crontab -l 2>/dev/null | grep -v hourly_monitor > /tmp/cron_tmp || true; echo '5 * * * * cd ~/wti && set -a && source .env && set +a && PYTHONPATH=~/wti /usr/bin/python3 warehouse/monitoring/hourly_monitor.py >> ~/wti/logs/hourly_monitor.log 2>&1' >> /tmp/cron_tmp; crontab /tmp/cron_tmp; rm /tmp/cron_tmp"

Write-Host "[OK] Cron job configured" -ForegroundColor Green

# ============================================================
# Done
# ============================================================
Write-Host ""
Write-Host "========================================"
Write-Host "  Deployment Complete!" -ForegroundColor Green
Write-Host "========================================"
Write-Host ""
Write-Host "EC2 IP: $EC2_IP"
Write-Host "Project: ~/wti"
Write-Host "Log: ~/wti/logs/hourly_monitor.log"
Write-Host "Schedule: Every hour at :05"
Write-Host ""
Write-Host "To check logs manually:"
Write-Host "ssh -i $KeyPath $RemoteHost `"tail -f ~/wti/logs/hourly_monitor.log`""