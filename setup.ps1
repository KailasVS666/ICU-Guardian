# Install backend dependencies
Write-Host "📦 Installing backend dependencies..." -ForegroundColor Cyan
cd backend
pip install -r requirements.txt

# Install frontend dependencies
Write-Host "`n📦 Installing frontend dependencies..." -ForegroundColor Cyan
cd ../frontend
npm install

# Return to root
cd ..

Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "`nTo start the application:" -ForegroundColor Yellow
Write-Host "  Backend:  cd backend && python main.py" -ForegroundColor White
Write-Host "  Frontend: cd frontend && npm run dev" -ForegroundColor White
Write-Host "`nOr use the start scripts:" -ForegroundColor Yellow
Write-Host "  .\start-backend.ps1" -ForegroundColor White
Write-Host "  .\start-frontend.ps1" -ForegroundColor White
