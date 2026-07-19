import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from database import db, User, PredictionHistory
from models.inference import load_dermascan_model, predict_image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_change_in_production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dermascan.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'images')
HEATMAP_FOLDER = os.path.join(UPLOAD_FOLDER, 'heatmaps')

# Ensure upload directories exist
for folder in [IMAGE_FOLDER, HEATMAP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

db.init_app(app)

# Load the model
app.model = load_dermascan_model(os.path.join(app.root_path, 'models', 'dermascan_model.h5'))

@app.before_request
def create_tables():
    if not hasattr(app, 'db_created'):
        db.create_all()
        app.db_created = True

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))
            
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to view the dashboard.', 'warning')
        return redirect(url_for('login'))
        
    history = PredictionHistory.query.filter_by(user_id=session['user_id']).order_by(PredictionHistory.timestamp.desc()).all()
    return render_template('dashboard.html', history=history)

@app.route('/upload', methods=['POST'])
def upload():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))
        
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('dashboard'))
        
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('dashboard'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Create a unique filename to prevent overwrites
        unique_id = str(uuid.uuid4())
        ext = filename.rsplit('.', 1)[1]
        unique_filename = f"{unique_id}.{ext}"
        
        filepath = os.path.join(IMAGE_FOLDER, unique_filename)
        file.save(filepath)
        
        # Predict
        heatmap_filename = f"heatmap_{unique_id}.jpeg"
        heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
        
        try:
            result = predict_image(filepath, app.model, heatmap_path)
            
            # Save history
            new_pred = PredictionHistory(
                user_id=session['user_id'],
                original_image_path=f"uploads/images/{unique_filename}",
                heatmap_image_path=f"uploads/heatmaps/{heatmap_filename}" if os.path.exists(heatmap_path) else None,
                class_name=result['class_name'],
                confidence=result['confidence'],
                risk_level=result['risk_level']
            )
            db.session.add(new_pred)
            db.session.commit()
            
            flash('Image successfully analyzed.', 'success')
            return redirect(url_for('result', result_id=new_pred.id))
            
        except Exception as e:
            flash(f'Error during prediction: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))
            
    flash('Invalid file format. Allowed formats: png, jpg, jpeg', 'danger')
    return redirect(url_for('dashboard'))

@app.route('/result/<int:result_id>')
def result(result_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    prediction = PredictionHistory.query.get_or_404(result_id)
    
    # Ensure they can only view their own results
    if prediction.user_id != session['user_id']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('dashboard'))
        
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
