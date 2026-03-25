# GitHub Push Instructions

Panduan lengkap untuk push project ke GitHub.

## Step 1: Buat Repository di GitHub

1. Buka https://github.com/new
2. Isi detail:
   - **Repository name**: `assistance_car` (atau nama pilihan Anda)
   - **Description**: "Computer vision perception system for autonomous driving assistance"
   - **Public/Private**: Pilih sesuai preferensi
   - **Initialize repository**: JANGAN centang (karena sudah ada .git lokal)

3. Klik "Create repository"

## Step 2: Konfigurasi Git User (Jika Belum)

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Atau untuk global config:
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Add Remote dan Push

Ganti `YOUR_USERNAME` dengan username GitHub Anda:

```bash
cd C:\Users\jofeb\Desktop\capstone_project\assistance_car

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/assistance_car.git

# Verify remote
git remote -v

# Add semua files (akan mengikuti .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: Assistance car perception system with YOLO, Lane Detection, and RealSense support"

# Push ke main branch
git branch -M main
git push -u origin main
```

## Step 4: Verify di GitHub

1. Buka https://github.com/YOUR_USERNAME/assistance_car
2. Pastikan semua files sudah ter-push (kecuali yang di .gitignore)
3. Check file size - jangan sampai > 100MB per file

## Troubleshooting

### Error: "fatal: not a git repository"
```bash
cd C:\Users\jofeb\Desktop\capstone_project\assistance_car
git status
```

### Error: "Permission denied (publickey)"
Perlu setup SSH key atau gunakan Personal Access Token:

```bash
# Setup via HTTPS dengan token
git remote remove origin
git remote add origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/assistance_car.git

# Setup via SSH (recommended)
# 1. Generate key: ssh-keygen -t ed25519 -C "your.email@example.com"
# 2. Add key ke GitHub Settings > SSH Keys
# 3. git remote add origin git@github.com:YOUR_USERNAME/assistance_car.git
```

### Large File Warning
Jika ada file > 100MB, gunakan Git LFS:
```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add Git LFS for large model files"
git push
```

## Post-Push Steps

1. **Add Topics** (GitHub repo page):
   - autonomous-driving
   - computer-vision
   - tensorflow
   - sterio-camera
   - lane-detection
   - real-time-processing

2. **Enable Discussions** (Settings > Features)

3. **Add CONTRIBUTING.md link** di Discussions

4. **Setup branch protection** (Settings > Branches):
   - Require pull request reviews
   - Require status checks

5. **Add GitHub Actions** (optional):
   - Automated tests
   - Code quality checks

## Git Workflow Ke Depan

```bash
# Feature branch
git checkout -b feature/nama-fitur
# ... edit files ...
git add .
git commit -m "feat: deskripsi fitur"
git push -u origin feature/nama-fitur
# Create Pull Request di GitHub

# Bug fix
git checkout -b fix/nama-bug
# ... edit files ...
git add .
git commit -m "fix: deskripsi bug"
git push -u origin fix/nama-bug
```

## Quick Reference

```bash
# Check status
git status

# Check logs
git log --oneline -10

# Sync dengan remote
git pull origin main

# View remote URL
git remote -v

# Change remote URL
git remote set-url origin https://github.com/NEW_USERNAME/assistance_car.git
```

Done! 🎉
