from flask import Flask, render_template, request, redirect, flash, jsonify
import mysql.connector
import os
import pickle

app = Flask(__name__)
app.secret_key = 'mysecretkey123'

# Use /tmp for uploads on cloud (Render), fallback to local
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_db_connection():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "tramway.proxy.rlwy.net"),
        port=int(os.environ.get("MYSQL_PORT", 35641)),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", "aLgTKfCvYPjSjvsQZHAzQCvKRqikTVCL"),
        database=os.environ.get("MYSQL_DATABASE", "railway")
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
            # Save photo to /tmp/uploads
            photo_filename = f"{name}_{os.urandom(4).hex()}.jpg"
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
            photo.save(photo_path)
            print(f"Photo saved at: {photo_path}")

            # NOTE: Face encoding is extracted by camera_recognition.py running locally.
            # Web app stores the photo path; encoding will be added by local script.
            # For now, store NULL encoding — camera script will update it.
            encoding_bytes = None

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

                print(f"Successfully registered: {name}")
                flash(f'{name} has been registered successfully! Please run the encoding script locally to enable recognition.', 'success')
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