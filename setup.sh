#!/bin/bash
echo "📦 Installing backend dependencies..."
cd backend
pip install -r requirements.txt

echo -e "\n📦 Installing frontend dependencies..."
cd ../frontend
npm install

cd ..
echo -e "\n✅ Setup complete!"
echo -e "\nTo start the application:"
echo "  Backend:  cd backend && python main.py"
echo "  Frontend: cd frontend && npm run dev"
