# 🧮 Gauss-Seidel Iterative Solver

A beautiful, modern web application for solving systems of linear equations using the Gauss-Seidel iterative method.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- 🎯 **Dual Input Modes**: Matrix form or equation form
- 📊 **Live Diagnostics**: Convergence analysis and residual calculations
- 💾 **Persistent History**: SQLite database stores all calculations
- 🎨 **Modern UI**: Beautiful gradients, animations, and responsive design
- 📈 **Iteration Tracking**: Detailed log of convergence process
- 🔍 **Stability Analysis**: Diagonal dominance checking

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd gauss-seidel-solver
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Open your browser**
```
http://127.0.0.1:5000
```

## 📦 Deployment

### Deploy to Render (Recommended - Free)

1. Fork this repository to your GitHub account
2. Go to [Render](https://render.com) and sign up
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
6. Click "Create Web Service"

Your app will be live in minutes! 🎉

### Deploy to Railway

1. Go to [Railway.app](https://railway.app)
2. Click "Start a New Project"
3. Select "Deploy from GitHub repo"
4. Select this repository
5. Railway will auto-detect and deploy

### Deploy to Heroku

```bash
heroku login
heroku create gauss-seidel-solver
git push heroku main
heroku open
```

## 📖 How to Use

### Matrix Form
1. Enter the number of variables
2. Fill in the coefficient matrix (A) and constants vector (b)
3. Adjust tolerance and max iterations if needed
4. Click "Compute Solution"

### Equation Form
1. Enter the number of equations
2. Type equations in standard form (e.g., `2x + 3y - z = 5`)
3. Variables can be any letters
4. Click "Compute Solution"

## 🧪 Example Problems

**Example 1: Simple 2x2 System**
```
4x + y = 11
x + 3y = 10
```
Solution: x = 2, y = 3

**Example 2: 3x3 System**
```
10x - y + 2z = 6
-x + 11y - z + 3w = 25
2x - y + 10z - w = -11
3x + 2y + w + 8z = 15
```

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: Set to `production` for deployment

### Customization

Edit `app.py` to modify:
- Tolerance default (line with `tolerance` input)
- Max iterations (line with `maxIterations`)
- Matrix size limits (currently 2-10)
- Database path

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Database**: SQLite
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Deployment**: Gunicorn

## 📊 API Endpoints

### `POST /solve`
Solve a system of linear equations

**Request Body**:
```json
{
  "mode": "matrix",
  "tol": 0.0001,
  "maxIter": 1000,
  "A": [[4, 1], [1, 3]],
  "b": [11, 10]
}
```

**Response**:
```json
{
  "converged": true,
  "iterCount": 15,
  "solution": [2.0, 3.0],
  "variables": ["x", "y"],
  "iterations": [...],
  "diagnostics": {...}
}
```

### `GET /history`
Retrieve calculation history

### `POST /history/clear`
Clear all history entries

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

MIT License - feel free to use this project for learning or production!

## 🙏 Credits

Created with ❤️ using Flask and modern web technologies.

## 📧 Support

If you encounter issues or have questions, please open an issue on GitHub.

---

**Happy solving! 🎓✨**