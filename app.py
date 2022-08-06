from flask import Flask, render_template, Response, request
import cv2
import test_model as tm
import os
import pyttsx3

camera = cv2.VideoCapture(0)
capture = 0

picsFolder = os.path.join('static', 'pics')


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


def gen_frames():
    global capture, result_val

    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            if capture:
                capture = 0
                cv2.imwrite('shots/shot.png', frame)
                result_val = tm.classify_waste(os.path.join('shots', 'shot.png'))
                print(result_val)
                engine = pyttsx3.init()
                if result_val == 'trash':
                    engine.say('This is trash please throw it in the garbage bin')
                elif result_val == 'biological':
                    engine.say('This is food waste, try to recycle')
                else:
                    engine.say(result_val + " please recycle this")
                engine.runAndWait()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera, result_val
    if request.method == 'POST':
        if request.form.get('click') == 'Capture & Detect':
            global capture
            capture=1

    if capture:
        if request.method == 'GET':
            return render_template('index.html')

        return render_template('index.html')














if __name__ == '__main__':
    app.run(debug = True)