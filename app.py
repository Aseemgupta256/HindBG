# app.py
import os
import time
import threading
import gc
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from models import db, User, ImageHistory
from bg_remover import BackgroundRemover
import uuid
from enhanced_processor import EnhancedProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join('static', 'processed')

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create background remover and enhanced processor
bg_remover = BackgroundRemover()
enhanced_processor = EnhancedProcessor(app.root_path)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return render_template('register.html')
        if existing_email:
            flash('Email already exists', 'danger')
            return render_template('register.html')
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password, tokens=5)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! You now have 5 free tokens.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    history = ImageHistory.query.filter_by(user_id=current_user.id).order_by(ImageHistory.created_at.desc()).all()
    return render_template('dashboard.html', history=history)

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    users = User.query.filter(User.id != current_user.id).all()
    return render_template('admin.html', users=users)

@app.route('/admin/add_tokens', methods=['POST'])
@login_required
def add_tokens():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Access denied'}), 403
    user_id = request.form.get('user_id')
    tokens = request.form.get('tokens', type=int)
    user = User.query.get(user_id)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    user.tokens += tokens
    db.session.commit()
    return jsonify({
        'success': True, 
        'message': f'Added {tokens} tokens to {user.username}',
        'newTotal': user.tokens
    })

@app.route('/admin/create_user', methods=['POST'])
@login_required
def create_user():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Access denied'}), 403
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    tokens = request.form.get('tokens', type=int, default=0)
    is_admin = request.form.get('is_admin') == 'on'
    existing_user = User.query.filter_by(username=username).first()
    existing_email = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({'success': False, 'message': 'Username already exists'}), 400
    if existing_email:
        return jsonify({'success': False, 'message': 'Email already exists'}), 400
    hashed_password = generate_password_hash(password)
    new_user = User(
        username=username, 
        email=email, 
        password=hashed_password, 
        tokens=tokens,
        is_admin=is_admin
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({
        'success': True, 
        'message': f'User {username} created successfully',
        'user': {
            'id': new_user.id,
            'username': new_user.username,
            'email': new_user.email,
            'tokens': new_user.tokens,
            'is_admin': new_user.is_admin
        }
    })

@app.route('/process', methods=['POST'])
@login_required
def process_image():
    if current_user.tokens <= 0:
        return jsonify({'success': False, 'message': 'You have no tokens left. Please contact admin to add more tokens.'}), 400
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'jpe', 'jfif', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'heif', 'heic'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'success': False, 'message': 'File type not allowed'}), 400

    # Cleanup previous session files
    for key in ['current_original', 'current_processed']:
        if key in session:
            try:
                path = os.path.join(app.root_path, 'static', session[key])
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                app.logger.error(f"Error removing previous file {session[key]}: {e}")
    
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    upload_path = os.path.join(upload_folder, filename)
    file.save(upload_path)
    
    try:
        processed_folder = os.path.join(app.root_path, 'static', 'processed')
        os.makedirs(processed_folder, exist_ok=True)
        
        # Check if enhanced processing is enabled via frontend form
        use_enhancement = request.form.get('use_enhancement', 'false').lower() == 'true'
        if use_enhancement:
            app.logger.info(f"Enhanced Processing Enabled for {filename}")
            enhanced_folder = os.path.join(app.root_path, 'static', 'temp_enhanced')
            os.makedirs(enhanced_folder, exist_ok=True)
            enhanced_path = enhanced_processor.enhance_image(upload_path, enhanced_folder, reality_ratio=0.5)
            app.logger.info("Removing background from enhanced image")
            processed_path = bg_remover.remove_background(enhanced_path, processed_folder)
            # Remove the intermediate enhanced file after processing
            if os.path.exists(enhanced_path) and enhanced_path != upload_path:
                os.remove(enhanced_path)
        else:
            app.logger.info(f"Enhanced Processing Disabled for {filename} - Using original upload")
            processed_path = bg_remover.remove_background(upload_path, processed_folder)
        
        # Do NOT delete the uploaded image here so that input preview remains.
        # The original file is retained in the uploads folder and will be deleted later via cleanup.
        
        # Update session paths for later cleanup or download
        original_relative = os.path.relpath(upload_path, os.path.join(app.root_path, 'static'))
        processed_relative = os.path.relpath(processed_path, os.path.join(app.root_path, 'static'))
        session['current_original'] = original_relative
        session['current_processed'] = processed_relative
        
        current_user.tokens -= 1
        db.session.commit()
        
        host_url = request.host_url.rstrip('/')
        original_url = f"{host_url}/static/{original_relative}"
        processed_url = f"{host_url}/static/{processed_relative}"
        
        # Start background thread for memory cleanup (invoking gc.collect after a short delay)
        cleanup_thread = threading.Thread(target=lambda: (time.sleep(2), gc.collect()))
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Image processed successfully',
            'original': original_url,
            'processed': processed_url,
            'download_url': url_for('download_file', filename=processed_relative),
            'remaining_tokens': current_user.tokens
        })
    except Exception as e:
        app.logger.error(f"Image processing error: {str(e)}")
        if os.path.exists(upload_path):
            os.remove(upload_path)
        gc.collect()
        error_message = str(e)
        if "RGBA" in error_message or "mode" in error_message:
            return jsonify({'success': False, 'message': 'Unsupported image format. Please try a different image.'}), 500
        elif "memory" in error_message.lower():
            return jsonify({'success': False, 'message': 'Image is too large to process. Please try a smaller image.'}), 500
        else:
            return jsonify({'success': False, 'message': 'Error processing image. Please try a different image.'}), 500

@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    try:
        file_path = os.path.join(app.root_path, 'static', filename)
        if not os.path.exists(file_path):
            flash('File not found', 'error')
            return redirect(url_for('dashboard'))
        download_name = os.path.basename(file_path)
        return send_file(file_path, as_attachment=True, download_name=download_name)
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

def clean_static_files():
    """
    Delete files in 'uploads' and 'processed' directories that are older than a specified threshold.
    By default, the threshold is set to 24 hours (24*3600 seconds).
    For testing, you may lower this value.
    """
    threshold = 24 * 3600  # 24 hours; adjust as needed for testing
    now = time.time()
    directories = [
        os.path.join(app.root_path, 'static', 'uploads'),
        os.path.join(app.root_path, 'static', 'processed')
    ]
    for directory in directories:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > threshold:
                    try:
                        os.remove(file_path)
                        app.logger.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        app.logger.warning(f"Failed to delete {file_path}: {e}")

@app.route('/cleanup_files', methods=['POST'])
def cleanup_files():
    # Remove session files first
    for key in ['current_original', 'current_processed']:
        if key in session:
            try:
                path = os.path.join(app.root_path, 'static', session[key])
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                app.logger.error(f"Error removing file {session[key]}: {e}")
            session.pop(key, None)
    # Run static file cleanup
    clean_static_files()
    return '', 204

@app.route('/admin/cleanup_static', methods=['POST'])
@login_required
def admin_cleanup_static():
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Access denied'}), 403
    clean_static_files()
    return jsonify({'success': True, 'message': 'Static files cleanup completed'})

with app.app_context():
    db.create_all()
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@example.com',
            password=generate_password_hash('admin123'),
            is_admin=True,
            tokens=999999
        )
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
