from flask import Flask, Response, request, abort, send_file
import os
from config import config

app = Flask(__name__)

# Path to the video file
VIDEO_FILE_PATH = config.DATA_DIR / 'nextcloud' / 'Fiets' / 'Raf_22-11-24_13_30.MP4'
# VIDEO_FILE_PATH = config.DATA_DIR / 'nextcloud' / 'Fiets' / 'stripped-gpmf-streams'/ 'frames_Raf_22-11-24_13_30.MP4'
# VIDEO_FILE_PATH = config.DATA_DIR / 'frames_cut-GH010039-0.2.mp4'
# VIDEO_FILE_PATH = config.DATA_DIR / 'cut-GH010039-0.2.mp4'

@app.route('/video')
def serve_video():
    # return send_file(VIDEO_FILE_PATH)
    try:
        if not os.path.exists(VIDEO_FILE_PATH):
            abort(404, description="Video file not found.")

        range_header = request.headers.get('Range', None)
        file_size = os.path.getsize(VIDEO_FILE_PATH)
        start, end = 0, file_size - 1  # Default range is the entire file

        if range_header:
            # Parse Range header
            range_match = range_header.replace('bytes=', '').split('-')
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else end

        # Validate range values
        start = max(0, start)
        end = min(end, file_size - 1)
        if start > end:
            abort(416, description="Requested Range Not Satisfiable")

        length = end - start + 1

        def generate():
            with open(VIDEO_FILE_PATH, 'rb') as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk_size = 1024 * 1024  # Stream in 1 MB chunks
                    data = f.read(min(chunk_size, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        response = Response(generate(), status=206, mimetype="video/mp4")
        response.headers.add("Content-Range", f"bytes {start}-{end}/{file_size}")
        response.headers.add("Accept-Ranges", "bytes")
        response.headers.add("Content-Length", str(length))
        response.headers.add("Connection", "keep-alive")  # Keep the connection open
        return response

    except Exception as e:
        app.logger.error(f"Error while serving video: {e}")
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True, threaded=True)




def get_chunk(filename, byte1=None, byte2=None):
    filesize = os.path.getsize(filename)
    yielded = 0
    yield_size = 1024 * 1024

    if byte1 is not None:
        if not byte2:
            byte2 = filesize
        yielded = byte1
        filesize = byte2

    with open(filename, 'rb') as f:
        content = f.read()

    while True:
        remaining = filesize - yielded
        if yielded == filesize:
            break
        if remaining >= yield_size:
            yield content[yielded:yielded+yield_size]
            yielded += yield_size
        else:
            yield content[yielded:yielded+remaining]
            yielded += remaining

@app.route('/')
def get_file():
    filename = str(config.DATA_DIR / 'cut-GH010039-0.2.mp4')
    filesize = os.path.getsize(filename)
    range_header = request.headers.get('Range', None)

    if range_header:
        byte1, byte2 = None, None
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()

        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])

        if not byte2:
            byte2 = byte1 + 1024 * 1024
            if byte2 > filesize:
                byte2 = filesize

        length = byte2 + 1 - byte1

        resp = Response(
            get_chunk(filename, byte1, byte2),
            status=206, mimetype='video/mp4',
            content_type='video/mp4',
            direct_passthrough=True
        )

        resp.headers.add('Content-Range',
                         'bytes {0}-{1}/{2}'
                         .format(byte1,
                                 length,
                                 filesize))
        return resp

    return Response(
        get_chunk(filename),
        status=200, mimetype='video/mp4'
    )