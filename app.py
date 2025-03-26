from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime
import cv2
import numpy as np
from main_no_gui import process_video
import threading
import json
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_db():
    db = sqlite3.connect('videos.db')
    db.row_factory = sqlite3.Row
    return db

# Database initialization
def init_db():
    db = get_db()
    c = db.cursor()
    
    # Simple videos table
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date DATETIME NOT NULL,
            status TEXT NOT NULL,
            output_path TEXT
        )
    ''')
    
    db.commit()
    db.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

@app.route('/')
def index():
    return render_template('index.html')

def generate_thumbnail(video_path, output_path=None):
    """Generate a thumbnail from a video file"""
    try:
        if not output_path:
            # Generate thumbnail path based on video path
            output_path = os.path.splitext(video_path)[0] + '_thumbnail.jpg'
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Seek to 1/3 of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames / 3))
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            # Try first frame if seeking failed
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                cap.release()
                return None
        
        # Save the frame as a thumbnail
        cv2.imwrite(output_path, frame)
        cap.release()
        
        return output_path
    except Exception as e:
        print(f"Error generating thumbnail: {str(e)}")
        return None

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Save to database
            db = get_db()
            c = db.cursor()
            c.execute('''
                INSERT INTO videos (filename, upload_date, status)
                VALUES (?, ?, ?)
            ''', (filename, datetime.now(), 'processing'))
            video_id = c.lastrowid
            db.commit()
            db.close()
            
            # Process video in background
            def process():
                try:
                    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{video_id}.mp4')
                    
                    # Generate thumbnail
                    thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], f'thumbnail_{video_id}.jpg')
                    
                    # Process the video
                    process_video(
                        filepath,
                        output_path,
                        model_path='yolov8n.pt',
                        conf_threshold=0.25
                    )
                    
                    # Generate thumbnail from processed video
                    generate_thumbnail(output_path, thumbnail_path)
                    
                    # Update database
                    db = get_db()
                    c = db.cursor()
                    c.execute('''
                        UPDATE videos 
                        SET status = ?, output_path = ?
                        WHERE id = ?
                    ''', ('completed', output_path, video_id))
                    db.commit()
                    db.close()
                    
                except Exception as e:
                    print(f"Error processing video: {str(e)}")
                    db = get_db()
                    c = db.cursor()
                    c.execute('''
                        UPDATE videos 
                        SET status = ?
                        WHERE id = ?
                    ''', ('error', video_id))
                    db.commit()
                    db.close()
            
            thread = threading.Thread(target=process)
            thread.start()
            
            return jsonify({
                'message': 'Video uploaded successfully',
                'video_id': video_id
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/status/<int:video_id>')
def get_status(video_id):
    db = get_db()
    c = db.cursor()
    c.execute('SELECT * FROM videos WHERE id = ?', (video_id,))
    result = c.fetchone()
    db.close()
    
    if result:
        return jsonify({
            'status': result['status'],
            'output_path': result['output_path']
        })
    return jsonify({'error': 'Video not found'}), 404

@app.route('/videos')
def list_videos():
    db = get_db()
    c = db.cursor()
    c.execute('SELECT * FROM videos ORDER BY upload_date DESC')
    videos = c.fetchall()
    db.close()
    
    return jsonify([{
        'id': v['id'],
        'filename': v['filename'],
        'upload_date': v['upload_date'],
        'status': v['status'],
        'output_path': v['output_path']
    } for v in videos])

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve video files with proper headers for video streaming"""
    try:
        # Create a response with the file
        video_path = os.path.join(os.getcwd(), filename)
        
        # Check if file exists
        if not os.path.isfile(video_path):
            return "Video file not found", 404
            
        # Get file size for content-length header
        file_size = os.path.getsize(video_path)
        
        # Handle range requests for video streaming
        range_header = request.headers.get('Range', None)
        if range_header:
            # Parse the range header
            bytes_range = range_header.replace('bytes=', '').split('-')
            start = int(bytes_range[0]) if bytes_range[0] else 0
            end = int(bytes_range[1]) if bytes_range[1] else file_size - 1
            
            # Create the response with range headers
            chunk_size = end - start + 1
            response = send_file(
                video_path,
                mimetype='video/mp4',
                conditional=True,
                add_etags=True,
                as_attachment=False
            )
            response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
            response.headers.add('Accept-Ranges', 'bytes')
            response.headers.add('Content-Length', chunk_size)
            response.status_code = 206  # Partial Content
            return response
        
        # Regular request (not range)
        return send_file(
            video_path,
            mimetype='video/mp4',
            conditional=True,
            add_etags=True,
            as_attachment=False
        )
    except Exception as e:
        print(f"Error serving video: {str(e)}")
        return "Error serving video", 500

@app.route('/thumbnail/<int:video_id>')
def serve_thumbnail(video_id):
    """Serve a video thumbnail image"""
    try:
        thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], f'thumbnail_{video_id}.jpg')
        
        # Check if thumbnail exists
        if not os.path.isfile(thumbnail_path):
            # Try to generate it if the video exists
            db = get_db()
            c = db.cursor()
            c.execute('SELECT output_path FROM videos WHERE id = ?', (video_id,))
            result = c.fetchone()
            db.close()
            
            if result and result['output_path']:
                video_path = result['output_path']
                if os.path.isfile(video_path):
                    generate_thumbnail(video_path, thumbnail_path)
            
            # If still doesn't exist, return default
            if not os.path.isfile(thumbnail_path):
                # Return a default image or 404
                return send_file('static/default_thumbnail.jpg') if os.path.exists('static/default_thumbnail.jpg') else ('Thumbnail not found', 404)
        
        return send_file(thumbnail_path, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error serving thumbnail: {str(e)}")
        return "Error serving thumbnail", 500

if __name__ == '__main__':
    app.run(debug=True) 