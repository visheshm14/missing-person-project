from flask import Flask, render_template, request, redirect, flash, jsonify
import mysql.connector
import os
import pickle
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = 'mysecretkey123'

# Smart upload folder — local when running locally, /tmp/uploads on Render
IS_LOCAL = os.path.exists('face_processor.py')
UPLOAD_FOLDER = 'uploads' if IS_LOCAL else '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load face processor only when running locally
face_processor = None
if IS_LOCAL:
    try:
        from face_processor import FaceProcessor
        face_processor = FaceProcessor()
        print("FaceProcessor loaded — running locally with encoding support")
    except Exception as e:
        print(f"FaceProcessor not available: {e}")
else:
    print("Running on cloud — encoding extraction disabled")


def get_db_connection():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST"),
        port=int(os.environ.get("MYSQL_PORT", 10261)),
        user=os.environ.get("MYSQL_USER"),
        password=os.environ.get("MYSQL_PASSWORD"),
        database=os.environ.get("MYSQL_DATABASE"),
        ssl_verify_cert=False,
        ssl_verify_identity=False,
        connection_timeout=30
    )


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register_missing_person():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        last_seen = request.form['last_seen_location']
        contact = request.form['contact_number']
        reporter = request.form['reporter_name']
        photo = request.files['photo']

        if photo:
            photo_filename = f"{name}_{os.urandom(4).hex()}.jpg"
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
            photo.save(photo_path)
            print(f"Photo saved at: {photo_path}")

            # Extract encoding if running locally
            encoding_bytes = None
            if face_processor is not None:
                print("Extracting face encoding locally...")
                face_encoding = face_processor.extract_face_encoding(photo_path)
                if face_encoding is not None:
                    encoding_bytes = pickle.dumps(face_encoding)
                    print(f"Encoding extracted! Length: {len(face_encoding)}")
                else:
                    flash('Could not extract face. Please use a clear front-facing photo.', 'error')
                    return redirect('/register')

            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO missing_persons 
                    (full_name, age, gender, last_seen_location, contact_number, reporter_name, photo_path, face_encoding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (name, age, gender, last_seen, contact, reporter, photo_path, encoding_bytes))
                conn.commit()
                cursor.close()
                conn.close()

                if encoding_bytes:
                    flash(f'{name} registered successfully with face encoding!', 'success')
                else:
                    flash(f'{name} registered! Run fix_encodings.py locally to add face encoding.', 'warning')
                return redirect('/')

            except Exception as e:
                print(f"Database error: {e}")
                flash(f'Database error: {str(e)}', 'error')
                return redirect('/register')
        else:
            flash('Please upload a photo.', 'error')
            return redirect('/register')

    return render_template('register.html')


@app.route('/list')
def list_missing():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM missing_persons WHERE status = 'missing'")
        persons = cursor.fetchall()
        cursor.close()
        conn.close()
        return render_template('list.html', persons=persons)
    except Exception as e:
        flash(f'Database error: {str(e)}', 'error')
        return redirect('/')


@app.route('/alerts')
def view_alerts():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT alerts.*, missing_persons.full_name 
            FROM alerts 
            JOIN missing_persons ON alerts.person_id = missing_persons.id
            ORDER BY detected_time DESC
        """)
        alerts = cursor.fetchall()
        cursor.close()
        conn.close()
        return render_template('alerts.html', alerts=alerts)
    except Exception as e:
        flash(f'Database error: {str(e)}', 'error')
        return redirect('/')


@app.route('/mark_found/<int:person_id>')
def mark_found(person_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE missing_persons SET status = 'found' WHERE id = %s",
            (person_id,)
        )
        conn.commit()
        cursor.close()
        conn.close()
        flash('Person marked as found!', 'success')
        return redirect('/list')
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect('/list')


@app.route('/stats')
def stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT COUNT(*) as total FROM missing_persons WHERE status = 'missing'")
        missing_count = cursor.fetchone()['total']

        cursor.execute("SELECT COUNT(*) as total FROM missing_persons WHERE status = 'found'")
        found_count = cursor.fetchone()['total']

        cursor.execute("SELECT COUNT(*) as total FROM alerts")
        alerts_count = cursor.fetchone()['total']

        cursor.close()
        conn.close()

        return jsonify({
            'missing': missing_count,
            'found': found_count,
            'alerts': alerts_count
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('stored_faces', exist_ok=True)
    app.run(debug=True)