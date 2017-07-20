from PIL import Image, ImageEnhance
import io
from google.cloud import vision
from google.cloud import language
from oauth2client.client import GoogleCredentials
import argparse
import os


credentials = GoogleCredentials.get_application_default()

vision_client = vision.Client.from_service_account_json('icr_tool.json')

language_client = language.Client()

def box_image(image_file):
    """
    Check out https://cloud.google.com/vision/docs/fulltext-annotations and
    https://cloud.google.com/vision/docs/detecting-fulltext#vision-document-text-detection-python
    for code/information I used
    """
    with open(image_file, 'rb') as image_file:
        content = image_file.read()
        image = vision_client.image(
            content=content)
        
        document = image.detect_full_text()
        boxes = []
        print(document.pages, document)
        print("running")
        for page in document.pages:
            print(page)
            print("ss")
            for block in page.blocks:
                block_words = []
                for paragraph in block.paragraphs:
                    block_words.extend(paragraph.words)

                block_symbols = []
                for word in block_words:
                    boxes.append(word.bounding_box)
                    block_symbols.extend(word.symbols)
           
        count = 0
        # Save words to a directory
        SAVE   = True
        filenames = []
        # Show words as we go
        SHOW   = False
        
        # Return list of cropped words
        RETURN = True
        pictures = []
        
        im = Image.open(image_file)
        try:
            sharpness = ImageEnhance.Sharpness(im)
            sharpy = sharpness.enhance(4)  
        except ValueError:
            print("Got ValueError when trying to sharpen, img is " + image_name)
            print("Proceeding without sharpening")
            sharpy = im
        #sharpy.show()

        gray_im = sharpy.convert('L')
        bw_im = gray_im.point(lambda x: 0 if x<80 else 255, '1')
        #bw_im.show()


        if SHOW:
            bw_im.show()
        for box in boxes:
            left  = min([box.vertices[i].x for i in range(4)]) - 20
            upper = min([box.vertices[i].y for i in range(4)]) - 0
            right = max([box.vertices[i].x for i in range(4)]) + 20 
            lower = max([box.vertices[i].y for i in range(4)]) + 0
            box = (left, upper, right, lower)
            region = bw_im.crop(box)
            if SHOW:
                region.show()
                raw_input("Enter to open another image, Ctrl+C to kill")
            if SAVE:
                img_name  = image_file.name.split('.')[0]
                extension = image_file.name.split('.')[1]
                print img_name, extension
                region.save(img_name+str(count)+'.'+extension)
                filenames.append(img_name+str(count)+'.'+extension)
                count += 1
            if RETURN:
                pictures.append(region)
        if RETURN:
            return (pictures, filenames)
