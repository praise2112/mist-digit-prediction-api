import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from keras.models import load_model
from numpy import newaxis
import numpy as np


from PIL import Image, ImageFilter


app = Flask(__name__)
CORS(app)
# CORS(app, support_credentials=True)



from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper

def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    # use str instead of basestring if using Python 3.x
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    # use str instead of basestring if using Python 3.x
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator


mnist_model = load_model("model/keras_mnist_2Layer_adam_128BS_20epochs.h5", compile=False)


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva



def crop(png_image_name):
     im = Image.open(png_image_name)
     print(im.size)
    # (364, 471)
     print(im.getbbox())
    # (64, 89, 278, 267)
     x, y, z, a = im.getbbox()
     im2 = im.crop((x-40, y-20, z+40, a+20))    # show have put im.getbbox() here as an argument, but we get better accuracy when there is a margin around
     print(im2.size)
    # (214, 178)
     im2.save("upload_folder/test.png")

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['POST', 'GET', 'OPTIONS'])
# @cross_origin(supports_credentials=True)
@crossdomain(origin='*')
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    print(filename)
    file.save(os.path.join('upload_folder', filename))

    virtualPath = "upload_folder/" + filename

    crop("upload_folder/" + filename)   # crop image to the size of the number. cropping the transparent spaces way from the png increases the accuracy because of better quality
    # crop save image at "upload_folder/test.png"

    img_ = Image.open("upload_folder/test.png")  # image extension *.png,*.jpg

    # img_ = img_.convert(mode='L', palette='P')  # convert image to black and white
    # img_ = Image.new("RGB", img_.size, (0, 0, 0))
    # img2_ = img_
    # img2_.save('test2after.png')

    # give the png image a white background color and resize to 28 by 28
    canvas = Image.new('RGBA', img_.size, (255, 255, 255, 255))  # Empty canvas colour (r,g,b,a)
    canvas.paste(img_, mask=img_)  # Paste the image onto the canvas, using it's alpha channel as mask
    canvas.thumbnail([28, 28], Image.ANTIALIAS)  # resize image
    canvas.save('temp/temp.png', format="PNG")

    # rezise the image again just for the hell of it
    img_ = Image.open('temp/temp.png')
    new_width = 28
    new_height = 28
    img_ = img_.resize((new_width, new_height), Image.ANTIALIAS)
    img_.save('temp/temp.png')  # format may what u want ,*.png,*jpg,*.gif

    # convert  the 28 x 28 image to mnist format
    x_ = imageprepare('temp/temp.png')  # file path here
    print(len(x_))  # mnist IMAGES are 28x28=784 pixels
    X_ = np.asarray(x_, dtype=np.float32)   # convert to numpy array

    # using our model to make predictions
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True
    predicted = mnist_model.predict_proba(X_[newaxis, ...])
    predicted2 = mnist_model.predict_classes(X_[newaxis, ...])

    # os.remove(virtualPath)

    print(predicted)
    result_ = dict()
    result_['prediction'] = predicted.tolist()
    result_['pred'] = predicted2.tolist()
    return result_


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80, threaded=False)