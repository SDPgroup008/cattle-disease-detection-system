# 🐄 Cattle Disease Detection System

An AI-powered web application for detecting cattle diseases using computer vision and machine learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Django](https://img.shields.io/badge/Django-4.2-green)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **🤖 AI Disease Detection**: Real-time cattle disease identification using ONNX models
- **📊 Analytics Dashboard**: Comprehensive statistics and visualizations
- **👥 User Management**: Secure authentication and role-based access
- **🐄 Cattle Registry**: Complete cattle information management
- **📱 Responsive Design**: Mobile-friendly interface
- **📈 History Tracking**: Detailed detection history and reports
- **🔍 Explainable AI**: Visual explanations for AI predictions

## 🚀 Live Demo

**[View Live Application](https://your-app-name.onrender.com)** *(Will be available after deployment)*

## 🛠️ Technology Stack

- **Backend**: Django 4.2, Django REST Framework
- **AI/ML**: ONNX Runtime, PyTorch, Scikit-learn
- **Database**: PostgreSQL
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap
- **Deployment**: Render.com
- **Storage**: WhiteNoise for static files

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL (for production)
- Git

## 🔧 Local Development Setup

1. **Clone the repository**
   \`\`\`bash
   git clone https://github.com/YOUR_USERNAME/cattle-disease-detection-system.git
   cd cattle-disease-detection-system
   \`\`\`

2. **Create virtual environment**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

3. **Install dependencies**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Set up database**
   \`\`\`bash
   python manage.py migrate
   python manage.py createsuperuser
   \`\`\`

5. **Run development server**
   \`\`\`bash
   python manage.py runserver
   \`\`\`

6. **Access the application**
   - Main app: http://127.0.0.1:8000
   - Admin panel: http://127.0.0.1:8000/admin

## 🌐 Deployment

This application is configured for easy deployment on Render.com:

1. **Fork this repository**
2. **Connect to Render**
3. **Deploy automatically** using the included `render.yaml`

Detailed deployment instructions: [DEPLOYMENT.md](DEPLOYMENT.md)

## 📊 Usage

1. **Register/Login** to access the system
2. **Upload cattle images** for disease detection
3. **View AI predictions** with confidence scores
4. **Register cattle** in the system database
5. **Monitor analytics** through the dashboard
6. **Export reports** for record keeping

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Django community for the excellent framework
- ONNX team for the runtime optimization
- Bootstrap for the responsive UI components
- Render.com for free hosting

## 📞 Support

If you have any questions or need help, please:
- Open an issue on GitHub
- Contact: your.email@example.com

---

⭐ **Star this repository if you found it helpful!**
